#ifndef HADROS_WITH_PYTHIA
#error "powheg_pythia_hadros_driver requires HADROS_WITH_PYTHIA=ON."
#endif

#include <Pythia8/Pythia.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::filesystem::path lhe;
    std::filesystem::path output;
    std::string interaction = "CC";
    int seed = 12345;
    int max_events = 0;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " --lhe pwgevents.lhe --output event_record.txt --interaction CC|NC"
        << " [--seed N] [--max-events N]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (key == "--lhe") {
            args.lhe = need_value(key);
        } else if (key == "--output") {
            args.output = need_value(key);
        } else if (key == "--interaction") {
            args.interaction = need_value(key);
        } else if (key == "--seed") {
            args.seed = std::stoi(need_value(key));
        } else if (key == "--max-events") {
            args.max_events = std::stoi(need_value(key));
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.lhe.empty() || args.output.empty()) {
        throw std::runtime_error("--lhe and --output are required.");
    }
    if (args.interaction != "CC" && args.interaction != "NC") {
        throw std::runtime_error("--interaction must be CC or NC.");
    }
    if (args.max_events < 0) {
        throw std::runtime_error("--max-events must be non-negative.");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (!std::filesystem::exists(args.lhe)) {
            throw std::runtime_error("LHE file not found: " + args.lhe.string());
        }
        std::filesystem::create_directories(args.output.parent_path());

        std::ofstream out(args.output);
        if (!out) {
            throw std::runtime_error("Could not open output event-record file: " + args.output.string());
        }
        out << std::setprecision(17);

        Pythia8::Pythia pythia("", false);
        pythia.readString("Print:quiet = on");
        pythia.readString("Init:showChangedSettings = off");
        pythia.readString("Init:showChangedParticleData = off");
        pythia.readString("Next:numberCount = 0");
        pythia.readString("Next:numberShowInfo = 0");
        pythia.readString("Next:numberShowProcess = 0");
        pythia.readString("Next:numberShowEvent = 0");
        pythia.readString("Random:setSeed = on");
        pythia.readString("Random:seed = " + std::to_string(args.seed));
        pythia.readString("Beams:frameType = 4");
        pythia.readString("Beams:LHEF = " + args.lhe.string());
        pythia.readString("HadronLevel:all = on");

        if (!pythia.init()) {
            throw std::runtime_error("PYTHIA8 failed to initialize the POWHEG LHE file.");
        }

        int accepted = 0;
        while ((args.max_events == 0 || accepted < args.max_events) && pythia.next()) {
            ++accepted;
            out << "HADROS_EVENT " << accepted << ' ' << pythia.info.weight()
                << ' ' << pythia.event.size() << "\n";
            for (int i = 0; i < pythia.event.size(); ++i) {
                const auto& p = pythia.event[i];
                out << "HADROS_PARTICLE "
                    << accepted << ' '
                    << i << ' '
                    << p.status() << ' '
                    << p.id() << ' '
                    << p.mother1() << ' '
                    << p.mother2() << ' '
                    << p.daughter1() << ' '
                    << p.daughter2() << ' '
                    << p.px() << ' '
                    << p.py() << ' '
                    << p.pz() << ' '
                    << p.e() << ' '
                    << p.m() << ' '
                    << p.charge() << ' '
                    << (p.isFinal() ? 1 : 0) << "\n";
            }
        }

        if (accepted == 0) {
            throw std::runtime_error("PYTHIA8 accepted zero events from the POWHEG LHE file.");
        }

        std::cout << "POWHEG_NUDIS_PYTHIA8 interaction=" << args.interaction
                  << " accepted_events=" << accepted
                  << " output=" << args.output << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << "powheg_pythia_hadros_driver failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
}
