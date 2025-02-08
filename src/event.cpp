#include "openmc/event.h"
#include "openmc/material.h"
#include "openmc/simulation.h"
#include "openmc/timer.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

SharedArray<EventQueueItem> calculate_fuel_xs_queue;
SharedArray<EventQueueItem> calculate_nonfuel_xs_queue;
SharedArray<EventQueueItem> advance_particle_queue;
SharedArray<EventQueueItem> surface_crossing_queue;
SharedArray<EventQueueItem> collision_queue;

vector<Particle> particles;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void init_event_queues(int64_t n_particles)
{
  simulation::calculate_fuel_xs_queue.reserve(n_particles);
  simulation::calculate_nonfuel_xs_queue.reserve(n_particles);
  simulation::advance_particle_queue.reserve(n_particles);
  simulation::surface_crossing_queue.reserve(n_particles);
  simulation::collision_queue.reserve(n_particles);

  simulation::particles.resize(n_particles);
}

void free_event_queues(void)
{
  simulation::calculate_fuel_xs_queue.clear();
  simulation::calculate_nonfuel_xs_queue.clear();
  simulation::advance_particle_queue.clear();
  simulation::surface_crossing_queue.clear();
  simulation::collision_queue.clear();

  simulation::particles.clear();
}

void dispatch_xs_event(int64_t buffer_idx)
{
  Particle& p = simulation::particles[buffer_idx];
  if (p.material() == MATERIAL_VOID ||
      !model::materials[p.material()]->fissionable()) {
    simulation::calculate_nonfuel_xs_queue.thread_safe_append({p, buffer_idx});
  } else {
    simulation::calculate_fuel_xs_queue.thread_safe_append({p, buffer_idx});
  }
}

void process_init_events(int64_t n_particles, int64_t source_offset)
{
  simulation::time_event_init.start();
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    initialize_history(simulation::particles[i], source_offset + i + 1);
    dispatch_xs_event(i);
  }

#ifdef OPENMC_USE_PUMIPIC
  if (settings::pumipic_on) {
    auto start_time = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(runtime)
    for (int64_t i = 0; i < n_particles; i++) {
      const auto particle_pos = simulation::particles[i].r();
      settings::particle_positions[i * 3 + 0] = particle_pos[0];
      settings::particle_positions[i * 3 + 1] = particle_pos[1];
      settings::particle_positions[i * 3 + 2] = particle_pos[2];
    }
    settings::particle_location_copy_time += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start_time)
                                               .count();

    settings::p_pumi_tally->initialize_particle_location(
      settings::particle_positions.data(),
      settings::max_particles_in_flight * 3);
  }
#endif

  simulation::time_event_init.stop();
}

void process_calculate_xs_events(SharedArray<EventQueueItem>& queue)
{
  simulation::time_event_calculate_xs.start();

  // TODO: If using C++17, perform a parallel sort of the queue
  // by particle type, material type, and then energy, in order to
  // improve cache locality and reduce thread divergence on GPU. Prior
  // to C++17, std::sort is a serial only operation, which in this case
  // makes it too slow to be practical for most test problems.
  //
  // std::sort(std::execution::par_unseq, queue.data(), queue.data() +
  // queue.size());

  int64_t offset = simulation::advance_particle_queue.size();
  ;

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < queue.size(); i++) {
    Particle* p = &simulation::particles[queue[i].idx];
    p->event_calculate_xs();

    // After executing a calculate_xs event, particles will
    // always require an advance event. Therefore, we don't need to use
    // the protected enqueuing function.
    simulation::advance_particle_queue[offset + i] = queue[i];
  }

  simulation::advance_particle_queue.resize(offset + queue.size());

  queue.resize(0);

  simulation::time_event_calculate_xs.stop();
}

void openmc_event_advance_wrapper()
{
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::advance_particle_queue.size(); i++) {
    int64_t buffer_idx = simulation::advance_particle_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    p.event_advance();
  }
}

#ifdef OPENMC_USE_PUMIPIC
void pumipic_event_advance_wrapper() {
  int64_t n_particles =
    std::min(settings::max_particles_in_flight, simulation::work_per_rank);
  // TODO: replace max_particles in flight with a more dynamic approach

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::advance_particle_queue.size(); i++) {
    int64_t buffer_idx = simulation::advance_particle_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    double distance = p.get_destination_distance();
    bool is_hit = false;
    p.is_hit_time_boundary(distance, is_hit);
    p.score_the_tallies(distance);
    p.set_particle_weight_to_zero_if_it_hit_time_boundary(is_hit);

    settings::particle_in_advance_queue[buffer_idx] = 1;
  }

  auto start_time = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(runtime)
  for (int64_t buffer_idx = 0; buffer_idx < n_particles; buffer_idx++) {
    Particle& p = simulation::particles[buffer_idx];
    const auto particle_pos = p.r();
    settings::particle_positions[buffer_idx * 3 + 0] = particle_pos[0];
    settings::particle_positions[buffer_idx * 3 + 1] = particle_pos[1];
    settings::particle_positions[buffer_idx * 3 + 2] = particle_pos[2];

    settings::particle_weights[buffer_idx] = p.wgt();
  }
  settings::particle_location_copy_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

  settings::p_pumi_tally->move_to_next_location(
    settings::particle_positions.data(), settings::particle_in_advance_queue.data(), settings::particle_weights.data(), settings::max_particles_in_flight*3);
}
#endif

void send_particles_to_other_queues();

void process_advance_particle_events()
{
  simulation::time_event_advance_particle.start();
#ifdef OPENMC_USE_PUMIPIC
  if (settings::pumipic_on) {
    pumipic_event_advance_wrapper();
  } else {
    openmc_event_advance_wrapper();
  }
#else
  openmc_event_advance_wrapper();
#endif
  send_particles_to_other_queues();

  simulation::time_event_advance_particle.stop();
}
void send_particles_to_other_queues()
{
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::advance_particle_queue.size(); i++) {
    int64_t buffer_idx = simulation::advance_particle_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    if (!p.alive())
      continue;
    if (p.collision_distance() > p.boundary().distance) {
      simulation::surface_crossing_queue.thread_safe_append({p, buffer_idx});
    } else {
      simulation::collision_queue.thread_safe_append({p, buffer_idx});
    }
  }

  simulation::advance_particle_queue.resize(0);
}

void process_surface_crossing_events()
{
  simulation::time_event_surface_crossing.start();

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::surface_crossing_queue.size(); i++) {
    int64_t buffer_idx = simulation::surface_crossing_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    p.event_cross_surface();
    p.event_revive_from_secondary();
    if (p.alive())
      dispatch_xs_event(buffer_idx);
  }

  simulation::surface_crossing_queue.resize(0);

  simulation::time_event_surface_crossing.stop();
}

void process_collision_events()
{
  simulation::time_event_collision.start();

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::collision_queue.size(); i++) {
    int64_t buffer_idx = simulation::collision_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    p.event_collide();
    p.event_revive_from_secondary();
    if (p.alive())
      dispatch_xs_event(buffer_idx);
  }

  simulation::collision_queue.resize(0);

  simulation::time_event_collision.stop();
}

void process_death_events(int64_t n_particles)
{
  simulation::time_event_death.start();
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle& p = simulation::particles[i];
    p.event_death();
  }
  simulation::time_event_death.stop();
}

} // namespace openmc
