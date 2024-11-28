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
// Particle: origin, destination, particle_id, wgt
typedef pumipic::MemberTypes<pumipic::Vector3d, pumipic::Vector3d, Omega_h::LO,
  Omega_h::LO>
  PPParticle;
typedef pumipic::ParticleStructure<PPParticle> PPPS;
typedef Kokkos::DefaultExecutionSpace PPExeSpace;

void init_pumi_libs(int& argc, char**& argv);

// Global Variables
extern PPPS* pumipic_ptcls;
extern pumipic::Library* pp_lib;
extern Omega_h::Library oh_lib;
}

#endif // OPENMC_PUMIPIC_PARTICLES_DATA_STRUCTURE_H
