#pragma once

#include "hadros/cascade/analytic_dis_backend.hpp"
#include "hadros/cascade/interaction_backend.hpp"

namespace hadros::cascade {

class NeutrinoDISInteractionAdapter final : public InteractionBackend {
public:
    explicit NeutrinoDISInteractionAdapter(double fixed_inelasticity = 0.25,
                                           std::uint64_t seed = 1,
                                           bool sample_inelasticity = false);

    std::string name() const override;
    InteractionResult interact(const PrimaryInteractionEvent& event) override;

private:
    AnalyticDISBackend backend_;
};

}  // namespace hadros::cascade
