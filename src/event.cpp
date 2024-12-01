#include "openmc/event.h"
#include "openmc/material.h"
#include "openmc/simulation.h"
#include "openmc/timer.h"
#ifdef OPENMC_USING_PUMIPIC
#include "pumipic_particles_data_structure.h"
#include <pumipic_adjacency.hpp>
#include <pumipic_adjacency.tpp>
#endif

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

#ifdef OPENMC_USING_PUMIPIC

void find_initial_elements_of_pumipic_ptcls(int64_t n_particles) {
  Omega_h::HostWrite<Omega_h::Real> h_particle_init_pos(3*n_particles, "particle initial position on host");
  Omega_h::Write<Omega_h::I8> h_part_in_adv_que(pumiinopenmc::pumi_ps_size, 0, "particle in advance queue init");
  Omega_h::HostWrite<Omega_h::I8> inAdvQueue(h_part_in_adv_que);

#pragma  omp parallel for schedule(runtime)
  for (int64_t i = 0; i<n_particles; i++) {
    const auto particle_pos = simulation::particles[i].r();
    h_particle_init_pos[i*3 + 0] = particle_pos[0];
    h_particle_init_pos[i*3 + 1] = particle_pos[1];
    h_particle_init_pos[i*3 + 2] = particle_pos[2];

    inAdvQueue[i] = 1;
  }
  Omega_h::Reals particle_init_pos(h_particle_init_pos);
  Omega_h::Read<Omega_h::I8> in_flight(h_part_in_adv_que);

  // assign particle next positions
  auto particle_dest = pumiinopenmc::pumipic_ptcls->get<1>();
  auto in_fly = pumiinopenmc::pumipic_ptcls->get<3>();
  auto set_particle_dest = PS_LAMBDA(const int &e, const int &pid, const int &mask){
    if (mask>0 && pid < n_particles){
      particle_dest(pid, 0) = particle_init_pos[pid*3+0];
      particle_dest(pid, 1) = particle_init_pos[pid*3+1];
      particle_dest(pid, 2) = particle_init_pos[pid*3+2];

      in_fly(pid) = in_flight[pid];
    }
  };
  pumipic::parallel_for(pumiinopenmc::pumipic_ptcls, set_particle_dest, "set initial position as dest to get initial elem");

  pumiinopenmc::search_and_rebuild(true);
}

#endif


void process_init_events(int64_t n_particles, int64_t source_offset)
{
  simulation::time_event_init.start();
#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    initialize_history(simulation::particles[i], source_offset + i + 1);
    dispatch_xs_event(i);
  }
  simulation::time_event_init.stop();

  // pumipic work
#ifdef OPENMC_USING_PUMIPIC
  find_initial_elements_of_pumipic_ptcls(n_particles);
#endif
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

#ifdef OPENMC_USING_PUMIPIC
void pumipic_event_advance_wrapper()
{
  // TODO rather than initializing every time, initialize once and use multiple times
  // TODO replace n_particles with max_particles_in_flight
  int64_t n_particles =
    std::min(settings::max_particles_in_flight, simulation::work_per_rank);
  Omega_h::HostWrite<Omega_h::Real> h_particle_dest_pos(
    3 * n_particles, "ptcl init position");
  Omega_h::Write<Omega_h::I8> h_part_in_adv_que_read(n_particles, 0, "ptcl in adv read");
  Omega_h::HostWrite<Omega_h::I8> h_particle_in_adv_queue(h_part_in_adv_que_read);

#pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < simulation::advance_particle_queue.size(); i++) {
    int64_t buffer_idx = simulation::advance_particle_queue[i].idx;
    Particle& p = simulation::particles[buffer_idx];
    double distance = p.get_destination_distance();
    p.is_hit_time_boundary(distance);
    p.score_the_tallies(distance);

    h_particle_in_adv_queue[buffer_idx] = 1;
  }

#pragma omp parallel for schedule(runtime)
  for (int64_t buffer_idx = 0; buffer_idx < n_particles; buffer_idx++) {
    Particle& p = simulation::particles[buffer_idx];
    const auto particle_pos = p.r();
    h_particle_dest_pos[buffer_idx * 3 + 0] = particle_pos[0];
    h_particle_dest_pos[buffer_idx * 3 + 1] = particle_pos[1];
    h_particle_dest_pos[buffer_idx * 3 + 2] = particle_pos[2];
  }

  Omega_h::Reals particle_init_pos(h_particle_dest_pos);
  Omega_h::Read<Omega_h::I8> in_flight(h_particle_in_adv_queue);
  // asssign next position to pumipic particles
  auto particle_dest = pumiinopenmc::pumipic_ptcls->get<1>();
  auto in_fly        = pumiinopenmc::pumipic_ptcls->get<3>();
  auto set_particle_dest =
    PS_LAMBDA(const int& e, const int& pid, const int& mask)
  {
    if (mask > 0) {
      in_fly (pid) = in_flight[pid];
      particle_dest(pid, 0) = particle_init_pos[3 * pid + 0];
      particle_dest(pid, 1) = particle_init_pos[3 * pid + 1];
      particle_dest(pid, 2) = particle_init_pos[3 * pid + 2];
    }
  };
  pumipic::parallel_for(
    pumiinopenmc::pumipic_ptcls, set_particle_dest, "set new destination");

  // now do the search
  pumiinopenmc::search_and_rebuild(false);
}

#endif

void process_advance_particle_events()
{
  simulation::time_event_advance_particle.start();
#ifdef OPENMC_USING_PUMIPIC
  pumipic_event_advance_wrapper();
#else
  openmc_event_advance_wrapper();
#endif

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

  simulation::time_event_advance_particle.stop();
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
