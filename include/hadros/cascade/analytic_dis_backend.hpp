#pragma once

#include <cstdint>
#include <random>

#include "hadros/cascade/event_generator_backend.hpp"

namespace hadros::cascade {

class AnalyticDISBackend final : public EventGeneratorBackend {
public:
    explicit AnalyticDISBackend(double fixed_inelasticity = 0.25,
                                std::uint64_t seed = 1,
                                bool sample_inelasticity = false);

    std::string name() const override;
    std::vector<SecondaryParticle> generate(const PrimaryNeutrinoEvent& event) override;

private:
    double draw_inelasticity();
    int charged_lepton_pdg_from_neutrino(int neutrino_pdg) const;

    double fixed_inelasticity_;
    bool sample_inelasticity_;
    std::mt19937_64 rng_;
};

}  // namespace hadros::cascade
