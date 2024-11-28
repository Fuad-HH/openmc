#include "pumipic_particles_data_structure.h"

namespace pumiinopenmc {
PPPS* pumipic_ptcls = nullptr; // Define the variable
pumipic::Library* pp_lib = nullptr;
Omega_h::Library oh_lib;

void init_pumi_libs(int& argc, char**& argv)
{
  pp_lib = new pumipic::Library(&argc, &argv);
  oh_lib = pp_lib->omega_h_lib();
}

} // pumiinopenmc