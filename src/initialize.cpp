#include "openmc/initialize.h"

#include <clocale>
#include <cstddef>
#include <cstdlib> // for getenv
#include <cstring>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/cross_sections.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/geometry_aux.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/memory.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/openmp_interface.h"
#include "openmc/output.h"
#include "openmc/plot.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/string_utils.h"
#include "openmc/summary.h"
#include "openmc/tallies/tally.h"
#include "openmc/thermal.h"
#include "openmc/timer.h"
#include "openmc/vector.h"
#include "openmc/weight_windows.h"
#ifdef OPENMC_USING_PUMIPIC
#include "pumipic_particles_data_structure.h"
#endif

#ifdef LIBMESH
#include "libmesh/libmesh.h"
#endif

#ifdef OPENMC_USING_PUMIPIC
#include "pumipic_particles_data_structure.h"

pumiinopenmc::PPPS* pp_create_particle_structure(Omega_h::Mesh mesh, pumipic::lid_t numPtcls){
  Omega_h::Int ne = mesh.nelems();
  pumiinopenmc::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  pumiinopenmc::PPPS::kkGidView element_gids("element_gids", ne);

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

  Omega_h::parallel_for(mesh.nelems(),
    OMEGA_H_LAMBDA(Omega_h::LO id){
      ptcls_per_elem[id] = (id==0) ? numPtcls : 0;
  });

#ifdef PP_USE_GPU // this flag is not added for cpu it will go though else which is okay
  printf("[INFO] Using GPU for simulation...");
  policy =
    Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#else
  openmc::write_message("PumiPIC Using CPU for simulation...\n");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#endif

  // needs pumipic with cabana
  pumiinopenmc::PPPS *ptcls =
    new pumipic::DPS<pumiinopenmc::PPParticle>(policy, ne, numPtcls, ptcls_per_elem, element_gids);

  return ptcls;
}

void start_pumi_particles_in_0th_element(Omega_h::Mesh& mesh, pumiinopenmc::PPPS* ptcls) {
  // find the centroid of the 0th element
  const auto& coords = mesh.coords();
  const auto& tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  Omega_h::Write<Omega_h::Real> centroid_of_el0(3, 0.0, "centroid");

  auto find_centroid_of_el0 = OMEGA_H_LAMBDA(Omega_h::LO id) {
    const auto nodes = Omega_h::gather_verts<4>(tet2node, id);
    Omega_h::Few<Omega_h::Vector<3>, 4> tet_node_coords = Omega_h::gather_vectors<4, 3>(coords, nodes);
    const auto centroid = o::average(tet_node_coords);
    centroid_of_el0[0] = centroid[0];
    centroid_of_el0[1] = centroid[1];
    centroid_of_el0[2] = centroid[2];
  };
  Omega_h::parallel_for(1, find_centroid_of_el0, "find centroid of element 0");

  // assign the location to all particles
  auto init_loc = ptcls->get<0>();
  auto pids     = ptcls->get<2>();

  auto set_initial_positions = PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask>0) {
      pids(pid) = pid;
      init_loc(pid, 0) = centroid_of_el0[0];
      init_loc(pid, 1) = centroid_of_el0[1];
      init_loc(pid, 2) = centroid_of_el0[2];
    }
  };
  pumipic::parallel_for(ptcls, set_initial_positions, "set initial particle positions");
}

void load_pumipic_mesh_and_init_particles(int& argc, char**& argv) {
  openmc::write_message(1, "Reading the Omega_h mesh " + openmc::settings::oh_mesh_fname + " to tally tracklength\n");
  pumiinopenmc::init_pumi_libs(argc, argv);

  if (openmc::settings::oh_mesh_fname.empty()){
    openmc::fatal_error("Omega_h mesh for PumiPIC is not given. Provide --ohMesh = <osh file>");
  }
  pumiinopenmc::full_mesh_ = Omega_h::binary::read(openmc::settings::oh_mesh_fname, &pumiinopenmc::oh_lib);
  if (pumiinopenmc::full_mesh_.dim() != 3){
    openmc::fatal_error("PumiPIC only works for 3D mesh now.\n");
  }
  Omega_h::LO nelems = pumiinopenmc::full_mesh_.nelems();
  openmc::write_message(1, "PumiPIC Loaded mesh " + openmc::settings::oh_mesh_fname + " with " +  std::to_string(nelems) + " elements...\n");

  long int real_n_particles = openmc::settings::n_particles;
  Omega_h::Write<Omega_h::LO> owners(nelems, 0, "owners");
  // all the particles are initialized in element 0 to do an initial search to find the starting locations
  // of the openmc given particles.
  pumiinopenmc::p_picparts_ = new pumipic::Mesh(pumiinopenmc::full_mesh_, Omega_h::LOs(owners));
  openmc::write_message(1, "PumiPIC mesh partitioned...\n");
  Omega_h::Mesh *mesh = pumiinopenmc::p_picparts_->mesh();
  pumiinopenmc::pumipic_ptcls = pp_create_particle_structure(*mesh, real_n_particles);
  start_pumi_particles_in_0th_element(*mesh, pumiinopenmc::pumipic_ptcls);
  pumiinopenmc::p_pumi_particle_at_elem_boundary_handler =
    std::make_unique<pumiinopenmc::PumiParticleAtElemBoundary>(mesh->nelems(),
      pumiinopenmc::pumipic_ptcls->capacity());

  openmc::write_message("PumiPIC Mesh and data structure created with "
                        + std::to_string(pumiinopenmc::p_picparts_->mesh()->nelems())
                        + " and " + std::to_string(pumiinopenmc::pumipic_ptcls->capacity())
                        + " as particle structure capacity\n");
}
#endif // OPENMC_USING_PUMIPIC


int openmc_init(int argc, char* argv[], const void* intracomm)
{
  using namespace openmc;

#ifdef OPENMC_MPI
  // Check if intracomm was passed
  MPI_Comm comm;
  if (intracomm) {
    comm = *static_cast<const MPI_Comm*>(intracomm);
  } else {
    comm = MPI_COMM_WORLD;
  }

  // Initialize MPI for C++
  initialize_mpi(comm);
#endif

  // Parse command-line arguments
  int err = parse_command_line(argc, argv);
  if (err)
    return err;

#ifdef OPENMC_USING_PUMIPIC
  load_pumipic_mesh_and_init_particles(argc, argv);
#endif

#ifdef LIBMESH
  const int n_threads = num_threads();
  // initialize libMesh if it hasn't been initialized already
  // (if initialized externally, the libmesh_init object needs to be provided
  // also)
  if (!settings::libmesh_init && !libMesh::initialized()) {
#ifdef OPENMC_MPI
    // pass command line args, empty MPI communicator, and number of threads.
    // Because libMesh was not initialized, we assume that OpenMC is the primary
    // application and that its main MPI comm should be used.
    settings::libmesh_init =
      make_unique<libMesh::LibMeshInit>(argc, argv, comm, n_threads);
#else
    // pass command line args, empty MPI communicator, and number of threads
    settings::libmesh_init =
      make_unique<libMesh::LibMeshInit>(argc, argv, 0, n_threads);
#endif

    settings::libmesh_comm = &(settings::libmesh_init->comm());
  }

#endif

  // Start total and initialization timer
  simulation::time_total.start();
  simulation::time_initialize.start();

#ifdef _OPENMP
  // If OMP_SCHEDULE is not set, default to a static schedule
  char* envvar = std::getenv("OMP_SCHEDULE");
  if (!envvar) {
    omp_set_schedule(omp_sched_static, 0);
  }
#endif

  // Initialize random number generator -- if the user specifies a seed, it
  // will be re-initialized later
  openmc::openmc_set_seed(DEFAULT_SEED);

  // Copy previous locale and set locale to C. This is a workaround for an issue
  // whereby when openmc_init is called from the plotter, the Qt application
  // framework first calls std::setlocale, which affects how pugixml reads
  // floating point numbers due to a bug:
  // https://github.com/zeux/pugixml/issues/469
  std::string prev_locale = std::setlocale(LC_ALL, nullptr);
  if (std::setlocale(LC_ALL, "C") == NULL) {
    fatal_error("Cannot set locale to C.");
  }

  // Read XML input files
  if (!read_model_xml())
    read_separate_xml_files();

  // Reset locale to previous state
  if (std::setlocale(LC_ALL, prev_locale.c_str()) == NULL) {
    fatal_error("Cannot reset locale.");
  }

  // Write some initial output under the header if needed
  initial_output();

  // Check for particle restart run
  if (settings::particle_restart_run)
    settings::run_mode = RunMode::PARTICLE;

  // Stop initialization timer
  simulation::time_initialize.stop();
  simulation::time_total.stop();

  return 0;
}

namespace openmc {

#ifdef OPENMC_MPI
void initialize_mpi(MPI_Comm intracomm)
{
  mpi::intracomm = intracomm;

  // Initialize MPI
  int flag;
  MPI_Initialized(&flag);
  if (!flag)
    MPI_Init(nullptr, nullptr);

  // Determine number of processes and rank for each
  MPI_Comm_size(intracomm, &mpi::n_procs);
  MPI_Comm_rank(intracomm, &mpi::rank);
  mpi::master = (mpi::rank == 0);

  // Create bank datatype
  SourceSite b;
  MPI_Aint disp[10];
  MPI_Get_address(&b.r, &disp[0]);
  MPI_Get_address(&b.u, &disp[1]);
  MPI_Get_address(&b.E, &disp[2]);
  MPI_Get_address(&b.time, &disp[3]);
  MPI_Get_address(&b.wgt, &disp[4]);
  MPI_Get_address(&b.delayed_group, &disp[5]);
  MPI_Get_address(&b.surf_id, &disp[6]);
  MPI_Get_address(&b.particle, &disp[7]);
  MPI_Get_address(&b.parent_id, &disp[8]);
  MPI_Get_address(&b.progeny_id, &disp[9]);
  for (int i = 9; i >= 0; --i) {
    disp[i] -= disp[0];
  }

  int blocks[] {3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
  MPI_Datatype types[] {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
    MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_LONG, MPI_LONG};
  MPI_Type_create_struct(10, blocks, disp, types, &mpi::source_site);
  MPI_Type_commit(&mpi::source_site);
}
#endif // OPENMC_MPI

int parse_command_line(int argc, char* argv[])
{
  int last_flag = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg {argv[i]};
    if (arg[0] == '-') {
      if (arg == "-p" || arg == "--plot") {
        settings::run_mode = RunMode::PLOTTING;
        settings::check_overlaps = true;

      } else if (arg == "-n" || arg == "--particles") {
        i += 1;
        settings::n_particles = std::stoll(argv[i]);

      } else if (arg == "-e" || arg == "--event") {
        settings::event_based = true;
      } else if (arg == "-r" || arg == "--restart") {
        i += 1;
        // Check what type of file this is
        hid_t file_id = file_open(argv[i], 'r', true);
        std::string filetype;
        read_attribute(file_id, "filetype", filetype);
        file_close(file_id);

        // Set path and flag for type of run
        if (filetype == "statepoint") {
          settings::path_statepoint = argv[i];
          settings::path_statepoint_c = settings::path_statepoint.c_str();
          settings::restart_run = true;
        } else if (filetype == "particle restart") {
          settings::path_particle_restart = argv[i];
          settings::particle_restart_run = true;
        } else {
          auto msg =
            fmt::format("Unrecognized file after restart flag: {}.", filetype);
          strcpy(openmc_err_msg, msg.c_str());
          return OPENMC_E_INVALID_ARGUMENT;
        }

        // If its a restart run check for additional source file
        if (settings::restart_run && i + 1 < argc) {
          // Check if it has extension we can read
          if (ends_with(argv[i + 1], ".h5")) {

            // Check file type is a source file
            file_id = file_open(argv[i + 1], 'r', true);
            read_attribute(file_id, "filetype", filetype);
            file_close(file_id);
            if (filetype != "source") {
              std::string msg {
                "Second file after restart flag must be a source file"};
              strcpy(openmc_err_msg, msg.c_str());
              return OPENMC_E_INVALID_ARGUMENT;
            }

            // It is a source file
            settings::path_sourcepoint = argv[i + 1];
            i += 1;

          } else {
            // Source is in statepoint file
            settings::path_sourcepoint = settings::path_statepoint;
          }

        } else {
          // Source is assumed to be in statepoint file
          settings::path_sourcepoint = settings::path_statepoint;
        }

      } else if (arg == "-g" || arg == "--geometry-debug") {
        settings::check_overlaps = true;
      } else if (arg == "-c" || arg == "--volume") {
        settings::run_mode = RunMode::VOLUME;
      } else if (arg == "-s" || arg == "--threads") {
        // Read number of threads
        i += 1;

#ifdef _OPENMP
        // Read and set number of OpenMP threads
        int n_threads = std::stoi(argv[i]);
        if (n_threads < 1) {
          std::string msg {"Number of threads must be positive."};
          strcpy(openmc_err_msg, msg.c_str());
          return OPENMC_E_INVALID_ARGUMENT;
        }
        omp_set_num_threads(n_threads);
#else
        if (mpi::master) {
          warning("Ignoring number of threads specified on command line.");
        }
#endif
#ifdef OPENMC_USING_PUMIPIC
      } else if (arg == "--ohMesh") {
        settings::oh_mesh_fname = std::string(argv[i+1]);
        i += 1; // skip the file name
#endif

      } else if (arg == "-?" || arg == "-h" || arg == "--help") {
        print_usage();
        return OPENMC_E_UNASSIGNED;

      } else if (arg == "-v" || arg == "--version") {
        print_version();
        print_build_info();
        return OPENMC_E_UNASSIGNED;

      } else if (arg == "-t" || arg == "--track") {
        settings::write_all_tracks = true;

      } else {
        fmt::print(stderr, "Unknown option: {}\n", argv[i]);
        print_usage();
        return OPENMC_E_UNASSIGNED;
      }

      last_flag = i;
    }
  }

  // Determine directory where XML input files are
  if (argc > 1 && last_flag < argc - 1) {
    settings::path_input = std::string(argv[last_flag + 1]);

    // check that the path is either a valid directory or file
    if (!dir_exists(settings::path_input) &&
        !file_exists(settings::path_input)) {
      fatal_error(fmt::format(
        "The path specified to the OpenMC executable '{}' does not exist.",
        settings::path_input));
    }

    // Add slash at end of directory if it isn't there
    if (!ends_with(settings::path_input, "/") &&
        dir_exists(settings::path_input)) {
      settings::path_input += "/";
    }
  }

  return 0;
}

bool read_model_xml()
{
  std::string model_filename = settings::path_input;

  // if the current filename is a directory, append the default model filename
  if (model_filename.empty() || dir_exists(model_filename))
    model_filename += "model.xml";

  // if this file doesn't exist, stop here
  if (!file_exists(model_filename))
    return false;

  // try to process the path input as an XML file
  pugi::xml_document doc;
  if (!doc.load_file(model_filename.c_str())) {
    fatal_error(fmt::format(
      "Error reading from single XML input file '{}'", model_filename));
  }

  pugi::xml_node root = doc.document_element();

  // Read settings
  if (!check_for_node(root, "settings")) {
    fatal_error("No <settings> node present in the model.xml file.");
  }
  auto settings_root = root.child("settings");

  // Verbosity
  if (check_for_node(settings_root, "verbosity")) {
    settings::verbosity = std::stoi(get_node_value(settings_root, "verbosity"));
  }

  // To this point, we haven't displayed any output since we didn't know what
  // the verbosity is. Now that we checked for it, show the title if necessary
  if (mpi::master) {
    if (settings::verbosity >= 2)
      title();
  }

  write_message(
    fmt::format("Reading model XML file '{}' ...", model_filename), 5);

  read_settings_xml(settings_root);

  // If other XML files are present, display warning
  // that they will be ignored
  auto other_inputs = {"materials.xml", "geometry.xml", "settings.xml",
    "tallies.xml", "plots.xml"};
  for (const auto& input : other_inputs) {
    if (file_exists(settings::path_input + input)) {
      warning((fmt::format("Other XML file input(s) are present. These files "
                           "may be ignored in favor of the {} file.",
        model_filename)));
      break;
    }
  }

  // Read materials and cross sections
  if (!check_for_node(root, "materials")) {
    fatal_error(fmt::format(
      "No <materials> node present in the {} file.", model_filename));
  }

  if (settings::run_mode != RunMode::PLOTTING) {
    read_cross_sections_xml(root.child("materials"));
  }
  read_materials_xml(root.child("materials"));

  // Read geometry
  if (!check_for_node(root, "geometry")) {
    fatal_error(fmt::format(
      "No <geometry> node present in the {} file.", model_filename));
  }
  read_geometry_xml(root.child("geometry"));

  // Final geometry setup and assign temperatures
  finalize_geometry();

  // Finalize cross sections having assigned temperatures
  finalize_cross_sections();

  if (check_for_node(root, "tallies"))
    read_tallies_xml(root.child("tallies"));

  // Initialize distribcell_filters
  prepare_distribcell();

  if (check_for_node(root, "plots")) {
    read_plots_xml(root.child("plots"));
  } else {
    // When no <plots> element is present in the model.xml file, check for a
    // regular plots.xml file
    std::string filename = settings::path_input + "plots.xml";
    if (file_exists(filename)) {
      read_plots_xml();
    }
  }

  finalize_variance_reduction();

  return true;
}

void read_separate_xml_files()
{
  read_settings_xml();
  if (settings::run_mode != RunMode::PLOTTING) {
    read_cross_sections_xml();
  }
  read_materials_xml();
  read_geometry_xml();

  // Final geometry setup and assign temperatures
  finalize_geometry();

  // Finalize cross sections having assigned temperatures
  finalize_cross_sections();
  read_tallies_xml();

  // Initialize distribcell_filters
  prepare_distribcell();

  // Read the plots.xml regardless of plot mode in case plots are requested
  // via the API
  read_plots_xml();

  finalize_variance_reduction();
}

void initial_output()
{
  // write initial output
  if (settings::run_mode == RunMode::PLOTTING) {
    // Read plots.xml if it exists
    if (mpi::master && settings::verbosity >= 5)
      print_plot();

  } else {
    // Write summary information
    if (mpi::master && settings::output_summary)
      write_summary();

    // Warn if overlap checking is on
    if (mpi::master && settings::check_overlaps) {
      warning("Cell overlap checking is ON.");
    }
  }
}

} // namespace openmc
