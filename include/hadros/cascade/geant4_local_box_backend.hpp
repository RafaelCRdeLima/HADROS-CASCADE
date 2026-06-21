#pragma once

#include <string>
#include <vector>

#include "hadros/cascade/types.hpp"

namespace hadros::cascade {

struct Geant4LocalBoxOptions {
    double box_size_cm = 100.0;
    double density_g_cm3 = 1.0;
    std::string material = "hydrogen";
    std::string physics_list = "FTFP_BERT";
    std::string energy_convention = "total";
    std::string safety_mode = "off";
    std::string uhe_transport_policy = "error";
    double geant4_hadron_max_kinetic_gev = 1.0e5;
    double geant4_lepton_max_kinetic_gev = 1.0e9;
    double geant4_photon_max_kinetic_gev = 1.0e9;
    double geant4_local_cm_per_rg = 1.0;
    bool one_particle_per_run = false;
    bool debug_single_particle = false;
};

struct Geant4LocalBoxEventResult {
    std::uint64_t event_id = 0;
    double input_energy_gev = 0.0;
    double deposited_energy_gev = 0.0;
    double escaped_energy_gev = 0.0;
    double invisible_energy_gev = 0.0;
    double untracked_energy_gev = 0.0;
    double unsupported_uhe_energy_gev = 0.0;
    double escaped_unsupported_uhe_energy_gev = 0.0;
    std::size_t n_unsupported_uhe_particles = 0;
    std::vector<SecondaryParticle> escaped_particles;
    std::vector<SecondaryParticle> unsupported_uhe_particles;
    std::string uhe_transport_policy = "error";
};

#ifdef HADROS_WITH_GEANT4
Geant4LocalBoxEventResult run_geant4_local_box_event(
    const std::vector<SecondaryParticle>& particles,
    const Geant4LocalBoxOptions& options);
#endif

}  // namespace hadros::cascade
