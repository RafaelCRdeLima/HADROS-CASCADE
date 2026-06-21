#include "hadros/cascade/neutrino_dis_interaction_adapter.hpp"

#include <stdexcept>

namespace hadros::cascade {

NeutrinoDISInteractionAdapter::NeutrinoDISInteractionAdapter(double fixed_inelasticity,
                                                             std::uint64_t seed,
                                                             bool sample_inelasticity)
    : backend_(fixed_inelasticity, seed, sample_inelasticity) {}

std::string NeutrinoDISInteractionAdapter::name() const {
    return "NeutrinoDISInteractionAdapter";
}

InteractionResult NeutrinoDISInteractionAdapter::interact(const PrimaryInteractionEvent& event) {
    if (!is_neutrino_pdg(event.primary.pdg_id)) {
        throw std::invalid_argument("NeutrinoDISInteractionAdapter received a non-neutrino primary.");
    }

    PrimaryNeutrinoEvent neutrino_event;
    neutrino_event.event_id = event.primary.event_id == 0 ? event.event_id : event.primary.event_id;
    neutrino_event.neutrino_pdg = event.primary.pdg_id;
    neutrino_event.energy_gev = event.primary.energy_gev;
    neutrino_event.weight = event.primary.weight;
    neutrino_event.charged_current = true;
    neutrino_event.interaction = event.point;

    auto secondaries = backend_.generate(neutrino_event);
    double visible = 0.0;
    double invisible = 0.0;
    for (const auto& particle : secondaries) {
        if (is_neutrino_pdg(particle.pdg_id == 0 ? particle.pdg : particle.pdg_id)) {
            invisible += particle.energy_gev;
        } else {
            visible += particle.energy_gev;
        }
    }

    InteractionResult result;
    result.event_id = neutrino_event.event_id;
    result.input_energy_gev = neutrino_event.energy_gev;
    result.visible_energy_gev = visible;
    result.invisible_energy_gev = invisible;
    result.escaped_energy_gev = invisible;
    result.deposited_energy_gev = visible;
    result.secondaries = secondaries;
    result.metadata = "analytic neutrino-DIS compatibility adapter";
    return result;
}

}  // namespace hadros::cascade
