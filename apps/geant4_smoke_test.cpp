#ifndef HADROS_WITH_GEANT4
#error "geant4_smoke_test requires HADROS_WITH_GEANT4=ON."
#endif

#include <FTFP_BERT.hh>
#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4RunManagerFactory.hh>
#include <G4SystemOfUnits.hh>
#include <G4UImanager.hh>
#include <G4VModularPhysicsList.hh>
#include <G4VUserDetectorConstruction.hh>
#include <QGSP_BERT.hh>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class SmokeDetector final : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* material = nist->FindOrBuildMaterial("G4_H");
        if (material == nullptr) {
            throw std::runtime_error("Could not construct G4_H material.");
        }

        auto* world_solid = new G4Box("smoke_world", 1.0 * cm, 1.0 * cm, 1.0 * cm);
        auto* world_logical = new G4LogicalVolume(world_solid, material, "smoke_world");
        return new G4PVPlacement(nullptr, {}, world_logical, "smoke_world", nullptr, false, 0);
    }
};

G4VModularPhysicsList* make_physics_list(const std::string& physics_name) {
    if (physics_name == "FTFP_BERT") {
        auto* physics = new FTFP_BERT;
        physics->SetVerboseLevel(0);
        return physics;
    }
    if (physics_name == "QGSP_BERT") {
        auto* physics = new QGSP_BERT;
        physics->SetVerboseLevel(0);
        return physics;
    }
    return nullptr;
}

void log_stage(const std::string& stage) {
    std::cout << "[geant4_smoke] " << stage << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string physics_name = argc >= 2 ? argv[1] : "FTFP_BERT";
    const bool run_initialize = argc < 3 || std::string(argv[2]) != "--no-initialize";

    try {
        log_stage("requested_physics_list=" + physics_name);

        log_stage("create_serial_run_manager:start");
        auto run_manager = std::unique_ptr<G4RunManager>(
            G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
        if (!run_manager) {
            throw std::runtime_error("G4RunManagerFactory returned nullptr.");
        }
        log_stage("create_serial_run_manager:ok");

        log_stage("construct_geometry_and_material:start");
        run_manager->SetUserInitialization(new SmokeDetector);
        log_stage("construct_geometry_and_material:registered");

        if (physics_name != "NONE") {
            log_stage("construct_physics_list:start");
            auto* physics = make_physics_list(physics_name);
            if (physics == nullptr) {
                throw std::runtime_error("Unknown physics list: " + physics_name);
            }
            run_manager->SetUserInitialization(physics);
            log_stage("construct_physics_list:registered");
        } else {
            log_stage("construct_physics_list:skipped");
        }

        if (run_initialize) {
            log_stage("run_initialize:start");
            run_manager->Initialize();
            log_stage("run_initialize:ok");
        } else {
            log_stage("run_initialize:skipped");
        }

        log_stage("ui_manager_available:start");
        auto* ui = G4UImanager::GetUIpointer();
        if (ui == nullptr) {
            throw std::runtime_error("G4UImanager pointer is null.");
        }
        log_stage("ui_manager_available:ok");

        std::cout << "GEANT4_SMOKE_TEST_OK physics_list=" << physics_name << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << "GEANT4_SMOKE_TEST_FAILED physics_list=" << physics_name
                  << " error=" << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
}
