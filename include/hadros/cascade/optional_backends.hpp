#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "hadros/cascade/cascade_backend.hpp"
#include "hadros/cascade/event_generator_backend.hpp"

namespace hadros::cascade {

#ifdef HADROS_WITH_PYTHIA
class PythiaShowerBackend final : public EventGeneratorBackend {
public:
    std::string name() const override { return "PythiaShowerBackend"; }
    std::vector<SecondaryParticle> generate(const PrimaryNeutrinoEvent&) override {
        throw std::runtime_error("PythiaShowerBackend is an optional shower stub; real PYTHIA coupling is not implemented in HADROS core.");
    }
};

class PowhegPythiaDISBackend final : public EventGeneratorBackend {
public:
    std::string name() const override { return "PowhegPythiaDISBackend"; }
    std::vector<SecondaryParticle> generate(const PrimaryNeutrinoEvent&) override {
        throw std::runtime_error("PowhegPythiaDISBackend is an optional DIS+shower stub; real external generator coupling is not implemented in HADROS core.");
    }
};
#endif

#ifdef HADROS_WITH_GEANT4
class Geant4LocalCascadeBackend final : public CascadeBackend {
public:
    std::string name() const override { return "Geant4LocalCascadeBackend"; }
    CascadeResult propagate(const std::vector<SecondaryParticle>&) override {
        throw std::runtime_error("Geant4LocalCascadeBackend is an optional local material-response stub; real GEANT4 coupling is not implemented in HADROS core.");
    }
};
#endif

#ifdef HADROS_WITH_HEPMC3
struct HepMC3ExchangeStub {
    static std::string name() { return "HepMC3ExchangeStub"; }
};
#endif

}  // namespace hadros::cascade
