//
// Created by Fuad Hasan on 11/27/24.
//

#ifndef OPENMC_PUMIPIC_PARTICLES_DATA_STRUCTURE_H
#define OPENMC_PUMIPIC_PARTICLES_DATA_STRUCTURE_H

#include <pumipic_library.hpp>
#include <pumipic_adjacency.hpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_adjacency.hpp>
#include <Omega_h_mesh.hpp>
#include <pumipic_mesh.hpp>

namespace pumiinopenmc {
// Particle: origin, destination, particle_id, in_advance_particle_queue
typedef pumipic::MemberTypes<pumipic::Vector3d, pumipic::Vector3d, Omega_h::LO, Omega_h::I16>
  PPParticle;
typedef pumipic::ParticleStructure<PPParticle> PPPS;
typedef Kokkos::DefaultExecutionSpace PPExeSpace;
struct PumiParticleAtElemBoundary;

void init_pumi_libs(int& argc, char**& argv);
void search_and_rebuild(bool initial);

// Global Variables
extern int64_t pumi_ps_size;
extern PPPS* pumipic_ptcls;
extern pumipic::Library* pp_lib;
extern Omega_h::Mesh full_mesh_;
extern pumipic::Mesh* p_picparts_;
extern Omega_h::Library oh_lib;
extern long double pumipic_tol;
extern std::unique_ptr<PumiParticleAtElemBoundary> p_pumi_particle_at_elem_boundary_handler;
extern Omega_h::Write<Omega_h::LO> elem_ids_;
extern Omega_h::Write<Omega_h::Real> inter_points_;
extern Omega_h::Write<Omega_h::LO> inter_faces_;


// other helper functions and structures

void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
  Omega_h::Write<Omega_h::LO> &ptcl_done,
  Omega_h::Write<Omega_h::LO> &lastExit);

struct PumiParticleAtElemBoundary {
  PumiParticleAtElemBoundary(Omega_h::LO nelems, Omega_h::LO capacity);

  void operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
    Omega_h::Write<Omega_h::LO> &inter_faces, Omega_h::Write<Omega_h::LO> &lastExit,
    Omega_h::Write<Omega_h::Real> &inter_points, Omega_h::Write<Omega_h::LO> &ptcl_done);

  void updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints);

  void evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> &xpoints);
  void finalizeAndWritePumiFlux(const std::string& filename);
  Omega_h::Reals normalizeFlux(Omega_h::Mesh &mesh);

  void mark_initial_as(bool initial);

  bool initial_; // in initial run, flux is not tallied
  Omega_h::Write<Omega_h::Real> flux_;
  Omega_h::Write<Omega_h::Real> prev_xpoint_;
};
}

#endif // OPENMC_PUMIPIC_PARTICLES_DATA_STRUCTURE_H
