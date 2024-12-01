#include "pumipic_particles_data_structure.h"
#include "openmc/mesh.h"
#include <Omega_h_shape.hpp>
#include <pumipic_ptcl_ops.hpp>
#include <Omega_h_file.hpp>

namespace pumiinopenmc {
//long double pumipic_tol = 1e-8;
int64_t pumi_ps_size = 100000;
PPPS* pumipic_ptcls = nullptr; // Define the variable
pumipic::Library* pp_lib = nullptr;
Omega_h::Library oh_lib;
Omega_h::Mesh full_mesh_;
pumipic::Mesh* p_picparts_ = nullptr;
std::unique_ptr<PumiParticleAtElemBoundary> p_pumi_particle_at_elem_boundary_handler = nullptr;
Omega_h::Write<Omega_h::LO> elem_ids_ = Omega_h::Write<Omega_h::LO> (0, "elem ids");
Omega_h::Write<Omega_h::Real> inter_points_ = Omega_h::Write<Omega_h::Real> (0, "intersection points");
Omega_h::Write<Omega_h::LO> inter_faces_ = Omega_h::Write<Omega_h::LO> (0, "intersection faces");

void init_pumi_libs(int& argc, char**& argv)
{
  pp_lib = new pumipic::Library(&argc, &argv);
  oh_lib = pp_lib->omega_h_lib();
}

void pp_move_to_new_element(Omega_h::Mesh &mesh, PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
  Omega_h::Write<Omega_h::LO> &ptcl_done,
  Omega_h::Write<Omega_h::LO> &lastExit) {
  const int dim = mesh.dim();
  const auto &face2elems = mesh.ask_up(dim - 1, dim);
  const auto &face2elemElem = face2elems.ab2b;
  const auto &face2elemOffset = face2elems.a2ab;
  const auto in_flight = ptcls->get<3>();

  auto set_next_element =
    PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && !ptcl_done[pid] && in_flight(pid)) {
      auto searchElm = elem_ids[pid];
      auto bridge = lastExit[pid];
      auto e2f_first = face2elemOffset[bridge];
      auto e2f_last = face2elemOffset[bridge + 1];
      auto upFaces = e2f_last - e2f_first;
      assert(upFaces == 2);
      auto faceA = face2elemElem[e2f_first];
      auto faceB = face2elemElem[e2f_first + 1];
      assert(faceA != faceB);
      assert(faceA == searchElm || faceB == searchElm);
      auto nextElm = (faceA == searchElm) ? faceB : faceA;
      elem_ids[pid] = nextElm;
    }
  };
  parallel_for(ptcls, set_next_element, "pumipic_set_next_element");
}

void apply_boundary_condition(Omega_h::Mesh &mesh, PPPS *ptcls,
  Omega_h::Write<Omega_h::LO> &elem_ids,
  Omega_h::Write<Omega_h::LO> &ptcl_done,
  Omega_h::Write<Omega_h::LO> &lastExit,
  Omega_h::Write<Omega_h::LO> &xFace) {

  // TODO: make this a member variable of the struct
  const auto &side_is_exposed = Omega_h::mark_exposed_sides(&mesh);

  auto checkExposedEdges =
    PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      assert(lastExit[pid] != -1);
      const Omega_h::LO bridge = lastExit[pid];
      const bool exposed = side_is_exposed[bridge];
      ptcl_done[pid] = exposed;
      xFace[pid] = lastExit[pid];
      //elem_ids[pid] = exposed ? -1 : elem_ids[pid];
    }
  };
  pumipic::parallel_for(ptcls, checkExposedEdges, "apply vacumm boundary condition");
}

PumiParticleAtElemBoundary::PumiParticleAtElemBoundary(Omega_h::LO nelems, Omega_h::LO capacity)
    : flux_(nelems, 0.0, "flux"),
      prev_xpoint_(capacity * 3, 0.0, "prev_xpoint"), initial_(true){
    printf(
      "[INFO] Particle handler at boundary with %d elements and %d "
      "x points size (3 * n_particles)\n",
      flux_.size(), prev_xpoint_.size());
  }

  void PumiParticleAtElemBoundary::operator()(Omega_h::Mesh &mesh, pumiinopenmc::PPPS *ptcls, Omega_h::Write<Omega_h::LO> &elem_ids,
    Omega_h::Write<Omega_h::LO> &inter_faces, Omega_h::Write<Omega_h::LO> &lastExit,
    Omega_h::Write<Omega_h::Real> &inter_points, Omega_h::Write<Omega_h::LO> &ptcl_done) {
    apply_boundary_condition(mesh, ptcls, elem_ids, ptcl_done, lastExit, inter_faces);
    pp_move_to_new_element(mesh, ptcls, elem_ids, ptcl_done, lastExit);
    if (!initial_) {
      evaluateFlux(ptcls, inter_points);
    }
  }

  void PumiParticleAtElemBoundary::mark_initial_as(bool initial)
  {
    initial_ = initial;
  }

  void PumiParticleAtElemBoundary::updatePrevXPoint(Omega_h::Write<Omega_h::Real> &xpoints) {
    OMEGA_H_CHECK_PRINTF(
      xpoints.size() <= prev_xpoint_.size() && prev_xpoint_.size() != 0,
      "xpoints size %d is greater than prev_xpoint size %d\n", xpoints.size(),
      prev_xpoint_.size());
    auto prev_xpoint = prev_xpoint_;
    auto update = OMEGA_H_LAMBDA(Omega_h::LO i) { prev_xpoint[i] = xpoints[i]; };
    Omega_h::parallel_for(xpoints.size(), update, "update previous xpoints");
  }

  void PumiParticleAtElemBoundary::evaluateFlux(PPPS *ptcls, Omega_h::Write<Omega_h::Real> &xpoints) {
    //Omega_h::Real total_particles = ptcls->nPtcls();
    auto prev_xpoint = prev_xpoint_;
    auto flux = flux_;
    auto in_flight = ptcls->get<3>();

    auto evaluate_flux =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
      if (mask > 0) {
        Omega_h::Vector<3> dest = {xpoints[pid * 3], xpoints[pid * 3 + 1],
          xpoints[pid * 3 + 2]};
        Omega_h::Vector<3> orig = {prev_xpoint[pid * 3], prev_xpoint[pid * 3 + 1],
          prev_xpoint[pid * 3 + 2]};

        Omega_h::Real parsed_dist = Omega_h::norm(dest - orig);  // / total_particles;
        Kokkos::atomic_add(&flux[e], parsed_dist * in_flight(pid));
      }
    };
    pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
  }

  Omega_h::Reals PumiParticleAtElemBoundary::normalizeFlux(Omega_h::Mesh &mesh) {
    const Omega_h::LO nelems = mesh.nelems();
    const auto &el2n = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
    const auto &coords = mesh.coords();

    auto flux = flux_;

    Omega_h::Write<Omega_h::Real> tet_volumes(flux_.size(), -1.0, "tet_volumes");
    Omega_h::Write<Omega_h::Real> normalized_flux(flux_.size(), -1.0, "normalized flux");

    auto normalize_flux_with_volume = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
      const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
      const auto elem_vert_coords = Omega_h::gather_vectors<4, 3>(coords, elem_verts);

      auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
      auto volume = Omega_h::simplex_size_from_basis(b);

      tet_volumes[elem_id] = volume;
      normalized_flux[elem_id] = flux[elem_id] / volume;
    };
    Omega_h::parallel_for(tet_volumes.size(), normalize_flux_with_volume,
      "normalize flux");

    mesh.add_tag(Omega_h::REGION, "volume", 1, Omega_h::Reals(tet_volumes));
    return Omega_h::Reals(normalized_flux);
  }

  void PumiParticleAtElemBoundary::finalizeAndWritePumiFlux(const std::string& filename){
    Omega_h::Mesh* mesh = p_picparts_->mesh();
    auto normalized_flux = pumiinopenmc::p_pumi_particle_at_elem_boundary_handler->normalizeFlux(
        *mesh);
    pumiinopenmc::full_mesh_.add_tag(Omega_h::REGION, "flux", 1, normalized_flux);
    Omega_h::vtk::write_parallel(filename, &pumiinopenmc::full_mesh_, 3);
  }

  void pumiUpdatePtclPositions(PPPS *ptcls) {
    auto x_ps_d = ptcls->get<0>();
    auto xtgt_ps_d = ptcls->get<1>();
    auto updatePtclPos = PS_LAMBDA(const int &, const int &pid, const bool &) {
      x_ps_d(pid, 0) = xtgt_ps_d(pid, 0);
      x_ps_d(pid, 1) = xtgt_ps_d(pid, 1);
      x_ps_d(pid, 2) = xtgt_ps_d(pid, 2);
      xtgt_ps_d(pid, 0) = 0.0;
      xtgt_ps_d(pid, 1) = 0.0;
      xtgt_ps_d(pid, 2) = 0.0;
    };
    ps::parallel_for(ptcls, updatePtclPos);
  }

  void pumiRebuild(pumipic::Mesh* picparts, PPPS *ptcls, Omega_h::Write<Omega_h::LO>& elem_ids) {
    pumiUpdatePtclPositions(ptcls);
    pumipic::migrate_lb_ptcls(*picparts, ptcls, elem_ids, 1.05);
    pumipic::printPtclImb(ptcls);
  }

  // search and update parent elements
  //! @param initial initial search finds the initial location of the particles and doesn't tally
  void search_and_rebuild(bool initial){
    p_pumi_particle_at_elem_boundary_handler->mark_initial_as(initial);
    Omega_h::LO maxLoops = 1000;
    auto orig = pumipic_ptcls->get<0>();
    auto dest = pumipic_ptcls->get<1>();
    auto pid  = pumipic_ptcls->get<2>();

    if (p_picparts_->mesh() == nullptr || p_picparts_->mesh()->nelems() == 0){
      openmc::fatal_error("PumiPIC meshes are not initialized properly\n");
    }

    bool isFoundAll = pumipic::particle_search(*p_picparts_->mesh(), pumipic_ptcls,
      orig, dest, pid, elem_ids_, inter_faces_,
      inter_points_, maxLoops, *p_pumi_particle_at_elem_boundary_handler);
    if (!isFoundAll){
      openmc::fatal_error("PumiPIC search Failed\n");
    }
    if (!initial) {
      pumiinopenmc::p_pumi_particle_at_elem_boundary_handler->updatePrevXPoint(
        inter_points_);
    }
    pumiRebuild(p_picparts_, pumipic_ptcls, elem_ids_);
  }

} // pumiinopenmc