#include "hadros/cascade/analytic_dis_backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace hadros::cascade {

AnalyticDISBackend::AnalyticDISBackend(double fixed_inelasticity,
                                       std::uint64_t seed,
                                       bool sample_inelasticity)
    : fixed_inelasticity_(fixed_inelasticity),
      sample_inelasticity_(sample_inelasticity),
      rng_(seed) {
    if (fixed_inelasticity_ <= 0.0 || fixed_inelasticity_ >= 1.0) {
        throw std::invalid_argument("AnalyticDISBackend inelasticity must be in (0, 1).");
    }
}

std::string AnalyticDISBackend::name() const {
    return "AnalyticDISBackend";
}

double AnalyticDISBackend::draw_inelasticity() {
    if (!sample_inelasticity_) {
        return fixed_inelasticity_;
    }
    std::uniform_real_distribution<double> y_distribution(0.05, 0.95);
    return y_distribution(rng_);
}

int AnalyticDISBackend::charged_lepton_pdg_from_neutrino(int neutrino_pdg) const {
    const int sign = neutrino_pdg >= 0 ? 1 : -1;
    const int abs_pdg = neutrino_pdg < 0 ? -neutrino_pdg : neutrino_pdg;
    if (abs_pdg == 12) {
        return sign * 11;
    }
    if (abs_pdg == 14) {
        return sign * 13;
    }
    if (abs_pdg == 16) {
        return sign * 15;
    }
    throw std::invalid_argument("AnalyticDISBackend expects neutrino PDG 12, 14, or 16.");
}

std::vector<SecondaryParticle> AnalyticDISBackend::generate(const PrimaryNeutrinoEvent& event) {
    if (event.energy_gev <= 0.0) {
        throw std::invalid_argument("AnalyticDISBackend received non-positive neutrino energy.");
    }
    if (!is_neutrino_pdg(event.neutrino_pdg)) {
        throw std::invalid_argument("AnalyticDISBackend expects a neutrino primary PDG code.");
    }

    const double y = draw_inelasticity();
    const double lepton_energy = (1.0 - y) * event.energy_gev;
    const double hadronic_energy = y * event.energy_gev;

    std::vector<SecondaryParticle> secondaries;
    secondaries.reserve(3);

    SecondaryParticle outgoing;
    outgoing.event_id = event.event_id;
    outgoing.parent_event_id = event.event_id;
    outgoing.pdg = event.charged_current ? charged_lepton_pdg_from_neutrino(event.neutrino_pdg)
                                         : event.neutrino_pdg;
    outgoing.pdg_id = outgoing.pdg;
    outgoing.energy_gev = lepton_energy;
    outgoing.pz_gev = lepton_energy;
    outgoing.weight = event.weight;
    outgoing.origin = event.charged_current ? "analytic_dis_cc_lepton"
                                            : "analytic_dis_nc_neutrino";
    outgoing.origin_backend = outgoing.origin;
    secondaries.push_back(outgoing);

    SecondaryParticle hadron_a;
    hadron_a.event_id = event.event_id;
    hadron_a.parent_event_id = event.event_id;
    hadron_a.pdg = 211;
    hadron_a.pdg_id = hadron_a.pdg;
    hadron_a.energy_gev = 0.6 * hadronic_energy;
    hadron_a.px_gev = hadron_a.energy_gev;
    hadron_a.weight = event.weight;
    hadron_a.origin = "analytic_dis_hadronic_proxy";
    hadron_a.origin_backend = hadron_a.origin;
    secondaries.push_back(hadron_a);

    SecondaryParticle hadron_b;
    hadron_b.event_id = event.event_id;
    hadron_b.parent_event_id = event.event_id;
    hadron_b.pdg = 2212;
    hadron_b.pdg_id = hadron_b.pdg;
    hadron_b.energy_gev = hadronic_energy - hadron_a.energy_gev;
    hadron_b.px_gev = -hadron_b.energy_gev;
    hadron_b.weight = event.weight;
    hadron_b.origin = "analytic_dis_hadronic_proxy";
    hadron_b.origin_backend = hadron_b.origin;
    secondaries.push_back(hadron_b);

    return secondaries;
}

}  // namespace hadros::cascade
