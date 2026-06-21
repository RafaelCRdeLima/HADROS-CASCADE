#pragma once

#include <string>

#include "hadros/cascade/types.hpp"

namespace hadros::cascade {

class InteractionBackend {
public:
    virtual ~InteractionBackend() = default;

    virtual std::string name() const = 0;
    virtual InteractionResult interact(const PrimaryInteractionEvent& event) = 0;
};

}  // namespace hadros::cascade
