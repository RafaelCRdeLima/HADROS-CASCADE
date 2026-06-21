#pragma once

#include <string>
#include <vector>

#include "hadros/cascade/types.hpp"

namespace hadros::cascade {

void write_interaction_points_jsonl(const std::string& path,
                                    const std::vector<InteractionPoint>& points);
std::vector<InteractionPoint> read_interaction_points_jsonl(const std::string& path);

void write_primary_particles_jsonl(const std::string& path,
                                   const std::vector<PrimaryParticle>& particles);
std::vector<PrimaryParticle> read_primary_particles_jsonl(const std::string& path);

void write_primary_interactions_jsonl(const std::string& path,
                                      const std::vector<PrimaryInteractionEvent>& events);
std::vector<PrimaryInteractionEvent> read_primary_interactions_jsonl(const std::string& path);

void write_primary_events_jsonl(const std::string& path,
                                const std::vector<PrimaryNeutrinoEvent>& events);
std::vector<PrimaryNeutrinoEvent> read_primary_events_jsonl(const std::string& path);

void write_secondaries_jsonl(const std::string& path,
                             const std::vector<SecondaryParticle>& particles);
std::vector<SecondaryParticle> read_secondaries_jsonl(const std::string& path);

void write_cascade_results_jsonl(const std::string& path,
                                 const std::vector<CascadeResult>& results);
std::vector<CascadeResult> read_cascade_results_jsonl(const std::string& path);

void write_cascade_energy_budget_csv(const std::string& path,
                                     const std::vector<CascadeResult>& results);

void write_interaction_results_jsonl(const std::string& path,
                                     const std::vector<InteractionResult>& results);
std::vector<InteractionResult> read_interaction_results_jsonl(const std::string& path);

}  // namespace hadros::cascade
