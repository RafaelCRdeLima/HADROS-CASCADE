#include "hadros/cascade/analytic_cascade_backend.hpp"

#include <stdexcept>

namespace hadros::cascade {

std::string AnalyticCascadeBackend::name() const {
    return "AnalyticCascadeBackend";
}

CascadeResult AnalyticCascadeBackend::propagate(const std::vector<SecondaryParticle>& secondaries) {
    CascadeResult result;
    if (!secondaries.empty()) {
        result.event_id = secondaries.front().event_id;
        result.weight = secondaries.front().weight;
    }

    for (const auto& particle : secondaries) {
        if (particle.energy_gev < 0.0) {
            throw std::invalid_argument("AnalyticCascadeBackend received negative secondary energy.");
        }

        if (is_neutrino_pdg(particle.pdg)) {
            result.escaped_neutrino_gev += particle.energy_gev;
            result.escaped_particles.push_back(particle);
        } else if (is_muon_pdg(particle.pdg)) {
            result.escaped_muon_gev += particle.energy_gev;
            result.escaped_particles.push_back(particle);
        } else if (is_electron_or_photon_pdg(particle.pdg)) {
            result.deposited_em_gev += particle.energy_gev;
        } else {
            result.deposited_hadronic_gev += particle.energy_gev;
        }
    }

    return result;
}

}  // namespace hadros::cascade
