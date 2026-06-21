#pragma once

#include "hadros/cascade/cascade_backend.hpp"

namespace hadros::cascade {

class AnalyticCascadeBackend final : public CascadeBackend {
public:
    std::string name() const override;
    CascadeResult propagate(const std::vector<SecondaryParticle>& secondaries) override;
};

}  // namespace hadros::cascade
