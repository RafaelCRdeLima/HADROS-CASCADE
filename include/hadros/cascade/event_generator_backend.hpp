#pragma once

#include <string>
#include <vector>

#include "hadros/cascade/types.hpp"

namespace hadros::cascade {

class EventGeneratorBackend {
public:
    virtual ~EventGeneratorBackend() = default;

    virtual std::string name() const = 0;
    virtual std::vector<SecondaryParticle> generate(const PrimaryNeutrinoEvent& event) = 0;
};

}  // namespace hadros::cascade
