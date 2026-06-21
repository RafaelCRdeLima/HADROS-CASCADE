#include "hadros/cascade/geant4_local_box_backend.hpp"

#ifdef HADROS_WITH_GEANT4

#include <FTFP_BERT.hh>
#include <G4Box.hh>
#include <G4Element.hh>
#include <G4Event.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4NistManager.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>
#include <G4ParticleTable.hh>
#include <G4ProcessVector.hh>
#include <G4PVPlacement.hh>
#include <G4RunManagerFactory.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4Types.hh>
#include <G4UserEventAction.hh>
#include <G4UserSteppingAction.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4VModularPhysicsList.hh>
#include <QGSP_BERT.hh>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace hadros::cascade {
namespace {

struct LocalAccumulator {
    double deposited_gev = 0.0;
    double escaped_gev = 0.0;
    std::vector<SecondaryParticle> escaped_particles;
};

bool supported_strict_pdg(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 22 || a == 11 || a == 13 || a == 211 || a == 321 ||
           a == 130 || a == 310 || a == 2212 || a == 2112;
}

bool is_hadron_for_uhe_policy(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 211 || a == 321 || a == 130 || a == 310 ||
           a == 2212 || a == 2112 || a >= 1000;
}

bool is_lepton_for_uhe_policy(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 11 || a == 13 || a == 15;
}

bool has_finite_four_momentum(const SecondaryParticle& particle) {
    return std::isfinite(particle.energy_gev) &&
           std::isfinite(particle.px_gev) &&
           std::isfinite(particle.py_gev) &&
           std::isfinite(particle.pz_gev) &&
           std::isfinite(particle.mass_gev);
}

double momentum2_gev2(const SecondaryParticle& particle) {
    return particle.px_gev * particle.px_gev +
           particle.py_gev * particle.py_gev +
           particle.pz_gev * particle.pz_gev;
}

double kinetic_energy_gev(const SecondaryParticle& particle, const Geant4LocalBoxOptions& options) {
    return options.energy_convention == "kinetic"
        ? particle.energy_gev
        : std::max(particle.energy_gev - particle.mass_gev, 0.0);
}

double geant4_kinetic_threshold_gev(int pdg, const Geant4LocalBoxOptions& options) {
    const int a = pdg < 0 ? -pdg : pdg;
    if (a == 22) {
        return options.geant4_photon_max_kinetic_gev;
    }
    if (is_lepton_for_uhe_policy(pdg)) {
        return options.geant4_lepton_max_kinetic_gev;
    }
    if (is_hadron_for_uhe_policy(pdg)) {
        return options.geant4_hadron_max_kinetic_gev;
    }
    return options.geant4_hadron_max_kinetic_gev;
}

bool exceeds_uhe_transport_limit(const SecondaryParticle& particle, const Geant4LocalBoxOptions& options) {
    const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
    const double kinetic = kinetic_energy_gev(particle, options);
    const double threshold = geant4_kinetic_threshold_gev(pdg, options);
    return std::isfinite(kinetic) && std::isfinite(threshold) && threshold >= 0.0 && kinetic > threshold;
}

bool strict_accept_particle(const SecondaryParticle& particle, const Geant4LocalBoxOptions& options) {
    const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
    if (is_neutrino_pdg(pdg)) {
        return false;
    }
    if (!supported_strict_pdg(pdg)) {
        return false;
    }
    if (!particle.stable) {
        return false;
    }
    if (!has_finite_four_momentum(particle) || particle.energy_gev < 0.0 || particle.mass_gev < 0.0) {
        return false;
    }
    if (momentum2_gev2(particle) <= 0.0) {
        return false;
    }
    if (options.energy_convention == "total" && particle.energy_gev + 1.0e-12 < particle.mass_gev) {
        return false;
    }
    if (options.energy_convention == "kinetic" && particle.energy_gev < 0.0) {
        return false;
    }
    return true;
}

class LocalBoxDetector final : public G4VUserDetectorConstruction {
public:
    explicit LocalBoxDetector(Geant4LocalBoxOptions options) : options_(std::move(options)) {}

    G4VPhysicalVolume* Construct() override {
        const double density = std::max(options_.density_g_cm3, 1.0e-30) * g / cm3;
        G4Material* material = nullptr;
        auto* nist = G4NistManager::Instance();
        if (options_.material == "water") {
            // Local benchmark material: H2O stoichiometry with configurable mass density.
            auto* hydrogen = nist->FindOrBuildElement("H");
            auto* oxygen = nist->FindOrBuildElement("O");
            material = new G4Material("HADROS_LocalWaterProxy", density, 2);
            material->AddElement(hydrogen, 2);
            material->AddElement(oxygen, 1);
        } else {
            auto* hydrogen = nist->FindOrBuildElement("H");
            material = new G4Material("HADROS_LocalHydrogenProxy", density, 1);
            material->AddElement(hydrogen, 1);
        }

        auto* world_material = nist->FindOrBuildMaterial("G4_Galactic");
        const auto world_half = 0.55 * options_.box_size_cm * cm;
        const auto box_half = 0.5 * options_.box_size_cm * cm;

        auto* world_solid = new G4Box("world", world_half, world_half, world_half);
        auto* world_logical = new G4LogicalVolume(world_solid, world_material, "world");
        auto* world_physical = new G4PVPlacement(nullptr, {}, world_logical, "world", nullptr, false, 0);

        auto* box_solid = new G4Box("local_box", box_half, box_half, box_half);
        auto* box_logical = new G4LogicalVolume(box_solid, material, "local_box");
        new G4PVPlacement(nullptr, {}, box_logical, "local_box", world_logical, false, 0);

        return world_physical;
    }

private:
    Geant4LocalBoxOptions options_;
};

class LocalPrimaryGenerator final : public G4VUserPrimaryGeneratorAction {
public:
    LocalPrimaryGenerator(std::vector<SecondaryParticle> particles, std::string energy_convention)
        : particles_(std::move(particles)),
          energy_convention_(std::move(energy_convention)),
          gun_(std::make_unique<G4ParticleGun>(1)) {}

    void GeneratePrimaries(G4Event* event) override {
        auto* table = G4ParticleTable::GetParticleTable();
        for (const auto& particle : particles_) {
            const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
            auto* definition = table->FindParticle(pdg);
            if (definition == nullptr) {
                continue;
            }
            const double px = particle.px_gev;
            const double py = particle.py_gev;
            const double pz = particle.pz_gev;
            G4ThreeVector direction(px, py, pz);
            if (direction.mag2() <= 0.0) {
                continue;
            }
            direction = direction.unit();
            gun_->SetParticleDefinition(definition);
            gun_->SetParticlePosition(G4ThreeVector(0.0, 0.0, 0.0));
            gun_->SetParticleMomentumDirection(direction);
            const double kinetic_gev = energy_convention_ == "kinetic"
                ? particle.energy_gev
                : std::max(particle.energy_gev - particle.mass_gev, 0.0);
            if (!std::isfinite(kinetic_gev) || kinetic_gev < 0.0) {
                continue;
            }
            gun_->SetParticleEnergy(kinetic_gev * GeV);
            gun_->GeneratePrimaryVertex(event);
        }
    }

private:
    std::vector<SecondaryParticle> particles_;
    std::string energy_convention_;
    std::unique_ptr<G4ParticleGun> gun_;
};

class LocalEventAction final : public G4UserEventAction {
public:
    explicit LocalEventAction(LocalAccumulator& accumulator) : accumulator_(accumulator) {}

    void BeginOfEventAction(const G4Event*) override {
        accumulator_.deposited_gev = 0.0;
        accumulator_.escaped_gev = 0.0;
        accumulator_.escaped_particles.clear();
    }

private:
    LocalAccumulator& accumulator_;
};

class LocalSteppingAction final : public G4UserSteppingAction {
public:
    LocalSteppingAction(LocalAccumulator& accumulator,
                        std::string energy_convention,
                        double geant4_local_cm_per_rg)
        : accumulator_(accumulator),
          energy_convention_(std::move(energy_convention)),
          geant4_local_cm_per_rg_(geant4_local_cm_per_rg) {}

    void UserSteppingAction(const G4Step* step) override {
        accumulator_.deposited_gev += step->GetTotalEnergyDeposit() / GeV;

        const auto* post = step->GetPostStepPoint();
        if (post == nullptr || post->GetStepStatus() != fWorldBoundary) {
            return;
        }
        const auto* track = step->GetTrack();
        if (track == nullptr) {
            return;
        }
        const double kinetic_gev = post->GetKineticEnergy() / GeV;
        if (!std::isfinite(kinetic_gev) || kinetic_gev < 0.0) {
            return;
        }
        const double mass_gev = track->GetDefinition()->GetPDGMass() / GeV;
        const double recorded_energy_gev = kinetic_gev;

        SecondaryParticle escaped;
        escaped.event_id = 0;
        escaped.parent_event_id = escaped.event_id;
        escaped.pdg = track->GetDefinition()->GetPDGEncoding();
        escaped.pdg_id = escaped.pdg;
        escaped.energy_gev = recorded_energy_gev;
        escaped.px_gev = track->GetMomentum().x() / GeV;
        escaped.py_gev = track->GetMomentum().y() / GeV;
        escaped.pz_gev = track->GetMomentum().z() / GeV;
        escaped.mass_gev = mass_gev;
        escaped.weight = track->GetWeight();
        escaped.stable = true;
        escaped.origin = "geant4_local_box_escape";
        escaped.origin_backend = "Geant4LocalBoxBackend";
        const auto position = post->GetPosition();
        escaped.geant4_box_origin_x_cm = 0.0;
        escaped.geant4_box_origin_y_cm = 0.0;
        escaped.geant4_box_origin_z_cm = 0.0;
        escaped.geant4_local_exit_x_cm = position.x() / cm;
        escaped.geant4_local_exit_y_cm = position.y() / cm;
        escaped.geant4_local_exit_z_cm = position.z() / cm;
        if (std::isfinite(geant4_local_cm_per_rg_) && geant4_local_cm_per_rg_ > 0.0) {
            escaped.geant4_local_cm_per_rg = geant4_local_cm_per_rg_;
            escaped.geant4_box_origin_x_rg = 0.0;
            escaped.geant4_box_origin_y_rg = 0.0;
            escaped.geant4_box_origin_z_rg = 0.0;
            escaped.exit_x_rg = escaped.geant4_local_exit_x_cm / geant4_local_cm_per_rg_;
            escaped.exit_y_rg = escaped.geant4_local_exit_y_cm / geant4_local_cm_per_rg_;
            escaped.exit_z_rg = escaped.geant4_local_exit_z_cm / geant4_local_cm_per_rg_;
        }
        escaped.position_status = "GEANT4_LOCAL_BOX_EXIT_POSITION_ONLY_NO_GLOBAL_KERR_ORIGIN";
        accumulator_.escaped_gev += recorded_energy_gev;
        accumulator_.escaped_particles.push_back(escaped);
    }

private:
    LocalAccumulator& accumulator_;
    std::string energy_convention_;
    double geant4_local_cm_per_rg_ = 1.0;
};

G4VModularPhysicsList* make_physics_list(const std::string& name) {
    if (name == "QGSP_BERT") {
        auto* physics = new QGSP_BERT;
        physics->SetVerboseLevel(0);
        return physics;
    }
    auto* physics = new FTFP_BERT;
    physics->SetVerboseLevel(0);
    return physics;
}

void print_particle_debug(
    const SecondaryParticle& particle,
    const Geant4LocalBoxOptions& options,
    const char* stage,
    bool allow_particle_table_lookup) {
    const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
    const double px = particle.px_gev;
    const double py = particle.py_gev;
    const double pz = particle.pz_gev;
    const double p2 = momentum2_gev2(particle);
    const double p = p2 > 0.0 ? std::sqrt(p2) : 0.0;
    const double kinetic_gev = options.energy_convention == "kinetic"
        ? particle.energy_gev
        : std::max(particle.energy_gev - particle.mass_gev, 0.0);
    std::cout << "[geant4-debug] stage=" << stage << "\n";
    std::cout << "[geant4-debug] event_id=" << particle.event_id << "\n";
    std::cout << "[geant4-debug] pdg=" << pdg << "\n";
    std::cout << "[geant4-debug] total_energy_gev=" << particle.energy_gev << "\n";
    std::cout << "[geant4-debug] mass_gev=" << particle.mass_gev << "\n";
    std::cout << "[geant4-debug] kinetic_energy_gev=" << kinetic_gev << "\n";
    std::cout << "[geant4-debug] momentum_gev=(" << px << "," << py << "," << pz << ")\n";
    if (p > 0.0 && std::isfinite(p)) {
        std::cout << "[geant4-debug] direction=(" << px / p << "," << py / p << "," << pz / p << ")\n";
    } else {
        std::cout << "[geant4-debug] direction=INVALID_ZERO_OR_NONFINITE\n";
    }
    std::cout << "[geant4-debug] material=" << options.material << "\n";
    std::cout << "[geant4-debug] density_g_cm3=" << options.density_g_cm3 << "\n";
    std::cout << "[geant4-debug] box_size_cm=" << options.box_size_cm << "\n";
    std::cout << "[geant4-debug] physics_list=" << options.physics_list << "\n";
    std::cout << "[geant4-debug] energy_convention=" << options.energy_convention << "\n";
    std::cout << "[geant4-debug] safety_mode=" << options.safety_mode << "\n";
    if (!allow_particle_table_lookup) {
        std::cout << "[geant4-debug] particle_table_lookup=DEFERRED_UNTIL_AFTER_INITIALIZE\n";
        return;
    }
    auto* table = G4ParticleTable::GetParticleTable();
    auto* definition = table ? table->FindParticle(pdg) : nullptr;
    std::cout << "[geant4-debug] particle_table_lookup=" << (definition ? "FOUND" : "MISSING") << "\n";
    if (!definition) {
        return;
    }
    std::cout << "[geant4-debug] particle_name=" << definition->GetParticleName() << "\n";
    std::cout << "[geant4-debug] geant4_pdg=" << definition->GetPDGEncoding() << "\n";
    std::cout << "[geant4-debug] geant4_mass_gev=" << definition->GetPDGMass() / GeV << "\n";
    std::cout << "[geant4-debug] geant4_charge_eplus=" << definition->GetPDGCharge() / eplus << "\n";
    auto* manager = definition->GetProcessManager();
    std::cout << "[geant4-debug] process_manager=" << (manager ? "FOUND" : "MISSING") << "\n";
    if (!manager) {
        return;
    }
    auto* processes = manager->GetProcessList();
    const int n_processes = processes ? processes->size() : 0;
    std::cout << "[geant4-debug] process_count=" << n_processes << "\n";
    for (int i = 0; processes && i < n_processes; ++i) {
        auto* process = (*processes)[i];
        if (!process) {
            continue;
        }
        std::cout << "[geant4-debug] process[" << i << "]=" << process->GetProcessName()
                  << " type=" << process->GetProcessType() << "\n";
    }
}

Geant4LocalBoxEventResult run_geant4_injected_once(
    const std::vector<SecondaryParticle>& injected,
    const Geant4LocalBoxOptions& options,
    std::uint64_t event_id) {
    Geant4LocalBoxEventResult result;
    result.event_id = event_id;
    if (injected.empty()) {
        return result;
    }

    LocalAccumulator accumulator;
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] run_manager_create_begin\n";
        for (const auto& particle : injected) {
            print_particle_debug(particle, options, "before_run_manager_initialize", false);
        }
    }
    auto run_manager = std::unique_ptr<G4RunManager>(
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] set_detector\n";
    }
    run_manager->SetUserInitialization(new LocalBoxDetector(options));
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] set_physics_list\n";
    }
    run_manager->SetUserInitialization(make_physics_list(options.physics_list));
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] set_primary_generator\n";
    }
    run_manager->SetUserAction(new LocalPrimaryGenerator(injected, options.energy_convention));
    run_manager->SetUserAction(new LocalEventAction(accumulator));
    run_manager->SetUserAction(new LocalSteppingAction(
        accumulator, options.energy_convention, options.geant4_local_cm_per_rg));
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] initialize_begin\n";
    }
    run_manager->Initialize();
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] initialize_done\n";
        for (const auto& particle : injected) {
            print_particle_debug(particle, options, "after_run_manager_initialize", true);
        }
        std::cout << "[geant4-debug] beam_on_begin\n";
    }
    run_manager->BeamOn(1);
    if (options.debug_single_particle) {
        std::cout << "[geant4-debug] beam_on_done\n";
    }

    result.deposited_energy_gev = accumulator.deposited_gev;
    result.escaped_energy_gev = accumulator.escaped_gev;
    result.escaped_particles = accumulator.escaped_particles;
    for (auto& escaped : result.escaped_particles) {
        escaped.event_id = result.event_id;
        escaped.parent_event_id = result.event_id;
    }
    return result;
}

}  // namespace

Geant4LocalBoxEventResult run_geant4_local_box_event(
    const std::vector<SecondaryParticle>& particles,
    const Geant4LocalBoxOptions& options) {
    Geant4LocalBoxEventResult result;
    if (!particles.empty()) {
        result.event_id = particles.front().event_id;
    }
    result.uhe_transport_policy = options.uhe_transport_policy;

    std::vector<SecondaryParticle> injected;
    const bool strict = options.safety_mode == "strict";
    for (const auto& particle : particles) {
        const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
        if (!std::isfinite(particle.energy_gev) || particle.energy_gev < 0.0) {
            throw std::runtime_error("GEANT4 local box received invalid input energy.");
        }
        if (options.energy_convention == "total" &&
            particle.energy_gev + 1.0e-12 < std::max(particle.mass_gev, 0.0)) {
            throw std::runtime_error("GEANT4 local box received total energy below rest mass.");
        }
        result.input_energy_gev += particle.energy_gev;
        if (is_neutrino_pdg(pdg)) {
            result.invisible_energy_gev += particle.energy_gev;
        } else if (exceeds_uhe_transport_limit(particle, options)) {
            if (options.uhe_transport_policy == "error") {
                throw std::runtime_error("GEANT4 UHE transport policy error: particle kinetic energy exceeds configured GEANT4 limit.");
            }
            if (options.uhe_transport_policy == "split_energy_proxy") {
                throw std::runtime_error("GEANT4 UHE transport policy split_energy_proxy is reserved but not implemented.");
            }
            if (options.uhe_transport_policy != "skip_to_escaped") {
                throw std::runtime_error("Unknown GEANT4 UHE transport policy: " + options.uhe_transport_policy);
            }
            SecondaryParticle skipped = particle;
            skipped.origin = "geant4_unsupported_uhe_policy";
            skipped.origin_backend = "geant4_unsupported_uhe_policy";
            result.unsupported_uhe_energy_gev += particle.energy_gev;
            result.escaped_unsupported_uhe_energy_gev += particle.energy_gev;
            result.n_unsupported_uhe_particles += 1;
            result.unsupported_uhe_particles.push_back(skipped);
        } else if (strict && !strict_accept_particle(particle, options)) {
            result.untracked_energy_gev += particle.energy_gev;
        } else {
            injected.push_back(particle);
        }
    }

    if (!injected.empty()) {
        if (options.one_particle_per_run) {
            for (const auto& particle : injected) {
                const auto one = run_geant4_injected_once({particle}, options, result.event_id);
                result.deposited_energy_gev += one.deposited_energy_gev;
                result.escaped_energy_gev += one.escaped_energy_gev;
                result.escaped_particles.insert(result.escaped_particles.end(), one.escaped_particles.begin(), one.escaped_particles.end());
            }
        } else {
            const auto all = run_geant4_injected_once(injected, options, result.event_id);
            result.deposited_energy_gev = all.deposited_energy_gev;
            result.escaped_energy_gev = all.escaped_energy_gev;
            result.escaped_particles = all.escaped_particles;
        }
    }

    const double accounted = result.deposited_energy_gev + result.escaped_energy_gev +
        result.invisible_energy_gev + result.escaped_unsupported_uhe_energy_gev;
    const double residual = result.input_energy_gev - accounted;
    if (std::isfinite(residual) && residual > result.untracked_energy_gev) {
        result.untracked_energy_gev = residual;
    }

    return result;
}

}  // namespace hadros::cascade

#endif
