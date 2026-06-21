#pragma once

#include <string>
#include <vector>

#include "hadros/cascade/types.hpp"

namespace hadros::cascade {

class CascadeBackend {
public:
    virtual ~CascadeBackend() = default;

    virtual std::string name() const = 0;
    virtual CascadeResult propagate(const std::vector<SecondaryParticle>& secondaries) = 0;
};

}  // namespace hadros::cascade
