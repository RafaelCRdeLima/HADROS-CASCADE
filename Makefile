# =========================================================
# HADROS-CASCADE Makefile
# Final scientific pipeline: UHE neutrino -> Kerr geodesic ->
# torus collision -> DIS -> PYTHIA -> GEANT4 -> backward camera
# =========================================================


# =========================================================
# Compiler and flags
# =========================================================

CXX := g++

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Iinclude -MMD -MP -fopenmp

LDFLAGS := -fopenmp

PYTHON ?= micromamba run -n dis python
CONFIG_PYTHON ?= python3


# =========================================================
# Directories
# =========================================================

SRC_DIR    := src
APP_DIR    := apps
SCRIPT_DIR := scripts
BUILD_DIR  := build
RUN_NAME ?=
ifeq ($(strip $(RUN_NAME)),)
OUTPUT_DIR := output
else
OUTPUT_DIR := output/$(RUN_NAME)
endif

FINAL_CONFIG ?= presets/config_web/final_pipeline_config.json
FINAL_CONFIG_WEB_HOST ?= 127.0.0.1
FINAL_CONFIG_WEB_PORT ?= 8877
CONFIG_PYTHON ?= python3


# =========================================================
# Runtime parameters
# =========================================================

ENU       ?= 1e5
ASPIN     ?= 0.0001
MBH_MSUN  ?= 3.0

CAM_R_OBS_RG ?= 60.0
CAM_THETA_DEG ?= 80.0
CAM_FOV_DEG   ?= 25.0
CAM_NX        ?= 100
CAM_NY        ?= 100
CAM_R_MAX_RG  ?= 120.0
CAM_STEP      ?= 0.001
GEODESIC_CACHE_PATH ?= $(OUTPUT_DIR)/rays/kerr_geodesics.bin

GEODESIC_PREVIEW_BIN ?= $(BUILD_DIR)/hadros_geodesic_preview

PREVIEW_OUTPUT_DIR   ?= $(OUTPUT_DIR)/camera_preview
PREVIEW_NX           ?= 256
PREVIEW_NY           ?= 144
PREVIEW_INTERACTIVE_NX ?= 256
PREVIEW_INTERACTIVE_NY ?= 144
PREVIEW_GEODESIC_MODEL ?= kerr_like
PREVIEW_NAV_MODE       ?= celestial_plus_torus_volume
PREVIEW_SKY_MODE       ?= interstellar_coordinate_grid
PREVIEW_R_MAX_RG       ?= 80
PREVIEW_OPAQUE_STRUCTURES ?= 0
PREVIEW_ALLOW_EXPENSIVE   ?= 0
PREVIEW_TORUS_MAX_ALPHA_STEP   ?= 0.055
PREVIEW_TORUS_EMISSIVITY_CUTOFF ?= 1e-8
PREVIEW_LIVE    ?= 1
PREVIEW_VSYNC   ?= 0
PREVIEW_QUALITY ?= medium

TORUS_RHO0     ?= 1.0e-2
TORUS_R0_RG    ?= 10.0
TORUS_SIGMA_RG ?= 5.0
TORUS_H_OVER_R ?= 0.25
DENSITY_PROFILE ?= gaussian
TORUS_RADIAL_POWER ?= 2.0
FUNNEL_DEPLETION   ?= 0.0
FUNNEL_THETA_DEG   ?= 15.0
ENVELOPE_RHO0      ?= 0.0
ENVELOPE_ALPHA     ?= 2.5
TORUS_R_MIN_RG     ?= 4.0
TORUS_R_MAX_RG     ?= 60.0
RHO_FLOOR          ?= 1.0e-99

NTHREADS ?= 4
OMP_NUM_THREADS ?= $(NTHREADS)
export OMP_NUM_THREADS

KERR_DERIVATIVE_MODE ?= finite_difference


# =========================================================
# Source groups
# =========================================================

KERR_SRC := \
	$(SRC_DIR)/kerr_metric.cpp \
	$(SRC_DIR)/kerr_geodesic.cpp \
	$(SRC_DIR)/kerr_camera.cpp

COMMON_SRC := \
	$(SRC_DIR)/sigma_table.cpp \
	$(SRC_DIR)/torus_profile.cpp


# =========================================================
# Object groups
# =========================================================

KERR_OBJS := \
	$(BUILD_DIR)/kerr_metric.o \
	$(BUILD_DIR)/kerr_geodesic.o \
	$(BUILD_DIR)/kerr_camera.o

COMMON_OBJS := \
	$(BUILD_DIR)/sigma_table.o \
	$(BUILD_DIR)/torus_profile.o


# =========================================================
# Phony targets
# =========================================================

.PHONY: all help build dirs clean clean-build clean-output \
	kerr-geodesics kerr-rays compute_kerr_geodesics dump_kerr_camera_rays \
	compute_kerr_particle_camera \
	compute_backward_camera_particle_channels \
	cascade_geant4_local_box geant4_smoke_test \
	compute_deposition_proxy_camera \
	hadros_camera_preview \
	build_geodesic_preview geodesic_preview \
	final-config-web final-pipeline-dry-run final-pipeline-run \
	build-plot-dashboard


# =========================================================
# Default target
# =========================================================

.DEFAULT_GOAL := help

all: help

help:
	@echo "HADROS-CASCADE targets:"
	@echo "  make build                         # compile all C++ binaries"
	@echo "  make kerr-geodesics                # trace and cache Kerr null geodesics"
	@echo "  make kerr-rays                     # dump Kerr camera rays for visualization"
	@echo "  make final-config-web              # start the cascade pipeline config web UI"
	@echo "  make final-pipeline-dry-run        # print the planned pipeline steps"
	@echo "  make final-pipeline-run            # run the full cascade pipeline"
	@echo "  make build-plot-dashboard          # build HTML dashboard from run outputs"
	@echo "  make cascade_geant4_local_box      # build GEANT4 local box binary (requires HADROS_WITH_GEANT4=ON)"
	@echo "  make geant4_smoke_test             # build GEANT4 smoke test (requires HADROS_WITH_GEANT4=ON)"
	@echo "  make clean                         # remove build artifacts and outputs"
	@echo ""
	@echo "Physics parameters:"
	@echo "  ASPIN     black hole spin (default: $(ASPIN))"
	@echo "  MBH_MSUN  black hole mass in solar masses (default: $(MBH_MSUN))"
	@echo "  ENU       neutrino energy in GeV (default: $(ENU))"
	@echo "  CAM_THETA_DEG  observer inclination in degrees (default: $(CAM_THETA_DEG))"
	@echo "  NTHREADS  OpenMP threads (default: $(NTHREADS))"


# =========================================================
# Directory setup
# =========================================================

dirs:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR)/rays
	mkdir -p $(OUTPUT_DIR)/particles


# =========================================================
# Generic compilation rules
# =========================================================

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(APP_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@


# =========================================================
# Kerr geodesic cache
# =========================================================

compute_kerr_geodesics: \
	$(KERR_OBJS) \
	$(BUILD_DIR)/compute_kerr_geodesics.o
	$(CXX) $(LDFLAGS) $^ -o $@

kerr-geodesics: dirs compute_kerr_geodesics
	KERR_DERIVATIVE_MODE=$(KERR_DERIVATIVE_MODE) OMP_NUM_THREADS=$(NTHREADS) \
	./compute_kerr_geodesics $(ASPIN) $(CAM_R_OBS_RG) $(CAM_THETA_DEG) \
	  $(CAM_FOV_DEG) $(CAM_NX) $(CAM_NY) $(CAM_R_MAX_RG) $(CAM_STEP) \
	  $(GEODESIC_CACHE_PATH)


# =========================================================
# Kerr camera ray dump (visualization)
# =========================================================

dump_kerr_camera_rays: \
	$(KERR_OBJS) \
	$(BUILD_DIR)/dump_kerr_camera_rays.o
	$(CXX) $(LDFLAGS) $^ -o $@

kerr-rays: dirs dump_kerr_camera_rays
	./dump_kerr_camera_rays $(ASPIN) $(CAM_R_OBS_RG) $(CAM_THETA_DEG) \
	  $(CAM_FOV_DEG) $(CAM_NX) $(CAM_NY) $(CAM_R_MAX_RG) $(CAM_STEP)


# =========================================================
# Geometric camera preview (no radiative transfer)
# =========================================================

hadros_camera_preview: dirs
	$(CXX) $(CXXFLAGS) \
	  $(APP_DIR)/hadros_camera_preview.cpp \
	  -o $(BUILD_DIR)/hadros_camera_preview


# =========================================================
# Geodesic preview (real Kerr raytracing, optional GLFW)
# =========================================================

build_geodesic_preview: dirs
	@if pkg-config --exists glfw3 2>/dev/null; then \
	  echo "[geodesic_preview] Building GLFW/OpenGL geodesic preview"; \
	  if $(CXX) $(CXXFLAGS) -DHADROS_GEODESIC_PREVIEW_GLFW \
	      $(APP_DIR)/hadros_geodesic_preview.cpp \
	      $(SRC_DIR)/kerr_metric.cpp $(SRC_DIR)/kerr_geodesic.cpp $(SRC_DIR)/kerr_camera.cpp \
	      -o $(GEODESIC_PREVIEW_BIN) $$(pkg-config --cflags --libs glfw3) -lGL; then \
	    echo "[geodesic_preview] GLFW/OpenGL build succeeded"; \
	  else \
	    echo "[geodesic_preview] GLFW/OpenGL build failed; building headless"; \
	    $(CXX) $(CXXFLAGS) \
	      $(APP_DIR)/hadros_geodesic_preview.cpp \
	      $(SRC_DIR)/kerr_metric.cpp $(SRC_DIR)/kerr_geodesic.cpp $(SRC_DIR)/kerr_camera.cpp \
	      -o $(GEODESIC_PREVIEW_BIN); \
	  fi; \
	else \
	  echo "[geodesic_preview] GLFW not found; building headless geodesic renderer"; \
	  $(CXX) $(CXXFLAGS) \
	    $(APP_DIR)/hadros_geodesic_preview.cpp \
	    $(SRC_DIR)/kerr_metric.cpp $(SRC_DIR)/kerr_geodesic.cpp $(SRC_DIR)/kerr_camera.cpp \
	    -o $(GEODESIC_PREVIEW_BIN); \
	fi

geodesic_preview: build_geodesic_preview
	mkdir -p $(PREVIEW_OUTPUT_DIR)
	HADROS_PREVIEW_OUTPUT_DIR=$(PREVIEW_OUTPUT_DIR) $(GEODESIC_PREVIEW_BIN) \
	  --nx $(PREVIEW_NX) --ny $(PREVIEW_NY) \
	  --spin $(ASPIN) \
	  --inclination $(CAM_THETA_DEG) \
	  --fov $(CAM_FOV_DEG) \
	  --r-obs $(CAM_R_OBS_RG) \
	  --r-max $(PREVIEW_R_MAX_RG) \
	  --mode $(PREVIEW_NAV_MODE) \
	  --quality $(PREVIEW_QUALITY)


# =========================================================
# Kerr particle-ray association camera (cascade-origin map, not full observer transport)
# =========================================================

compute_kerr_particle_camera: \
	$(KERR_OBJS) \
	$(BUILD_DIR)/compute_kerr_particle_camera.o
	$(CXX) $(LDFLAGS) $^ -o $(BUILD_DIR)/compute_kerr_particle_camera


# =========================================================
# Backward camera particle channels
# =========================================================

compute_backward_camera_particle_channels: \
	$(KERR_OBJS) \
	$(COMMON_OBJS) \
	$(BUILD_DIR)/compute_backward_camera_particle_channels.o
	$(CXX) $(LDFLAGS) $^ -o $(BUILD_DIR)/compute_backward_camera_particle_channels


# =========================================================
# GEANT4 local box (optional; requires HADROS_WITH_GEANT4=ON)
# =========================================================

CASCADE_SRC := \
	$(SRC_DIR)/cascade/jsonl_io.cpp \
	$(SRC_DIR)/cascade/geant4_local_box_backend.cpp \
	$(SRC_DIR)/cascade/analytic_cascade_backend.cpp \
	$(SRC_DIR)/cascade/analytic_dis_backend.cpp \
	$(SRC_DIR)/cascade/deposition_emissivity_field.cpp \
	$(SRC_DIR)/cascade/kerr_local_tetrad.cpp \
	$(SRC_DIR)/cascade/local_response_table.cpp \
	$(SRC_DIR)/cascade/neutrino_dis_interaction_adapter.cpp \
	$(SRC_DIR)/cascade/packet_kerr_null_propagator.cpp

cascade_geant4_local_box: dirs
	@if [ "$(HADROS_WITH_GEANT4)" != "ON" ]; then \
	  echo "cascade_geant4_local_box requires HADROS_WITH_GEANT4=ON"; \
	  exit 1; \
	fi
	$(CXX) $(CXXFLAGS) -DHADROS_WITH_GEANT4 \
	  $$(geant4-config --cflags) -std=c++17 \
	  $(APP_DIR)/cascade_geant4_local_box.cpp \
	  $(CASCADE_SRC) \
	  $(KERR_SRC) \
	  $(COMMON_SRC) \
	  -o $(BUILD_DIR)/cascade_geant4_local_box \
	  $$(geant4-config --libs)

geant4_smoke_test: dirs
	@if [ "$(HADROS_WITH_GEANT4)" != "ON" ]; then \
	  echo "geant4_smoke_test requires HADROS_WITH_GEANT4=ON"; \
	  exit 1; \
	fi
	$(CXX) $(CXXFLAGS) -DHADROS_WITH_GEANT4 \
	  $$(geant4-config --cflags) -std=c++17 \
	  $(APP_DIR)/geant4_smoke_test.cpp \
	  -o $(BUILD_DIR)/geant4_smoke_test \
	  $$(geant4-config --libs)


# =========================================================
# Deposition proxy camera (optional; requires HADROS_WITH_HDF5=ON)
# =========================================================

compute_deposition_proxy_camera: dirs
	@if [ "$(HADROS_WITH_HDF5)" != "ON" ]; then \
	  echo "compute_deposition_proxy_camera requires HADROS_WITH_HDF5=ON"; \
	  exit 1; \
	fi
	$(H5CXX) $(CXXFLAGS) -DHADROS_WITH_HDF5 \
	  $(APP_DIR)/compute_deposition_proxy_camera.cpp \
	  $(SRC_DIR)/kerr_metric.cpp \
	  $(SRC_DIR)/kerr_geodesic.cpp \
	  $(SRC_DIR)/kerr_camera.cpp \
	  -o $(BUILD_DIR)/compute_deposition_proxy_camera


# =========================================================
# Default CPU build
# =========================================================

build: dirs compute_kerr_geodesics dump_kerr_camera_rays \
	compute_kerr_particle_camera compute_backward_camera_particle_channels


# =========================================================
# Python workflow targets
# =========================================================

final-config-web:
	@echo "HADROS-CASCADE final scientific config web:"
	@echo "  Starting local server at http://$(FINAL_CONFIG_WEB_HOST):$(FINAL_CONFIG_WEB_PORT)"
	@echo "Press Ctrl+C to stop the server."
	$(CONFIG_PYTHON) $(SCRIPT_DIR)/config_web_final.py \
	  --host $(FINAL_CONFIG_WEB_HOST) \
	  --port $(FINAL_CONFIG_WEB_PORT) \
	  --config $(FINAL_CONFIG)

final-pipeline-dry-run:
	@if [ ! -f "$(FINAL_CONFIG)" ]; then \
	  $(CONFIG_PYTHON) $(SCRIPT_DIR)/config_web_final.py \
	    --write-default-pipeline-config $(FINAL_CONFIG); \
	fi
	$(CONFIG_PYTHON) $(SCRIPT_DIR)/run_hadros_final_pipeline.py \
	  $(FINAL_CONFIG) --dry-run

final-pipeline-run: build
	@if [ ! -f "$(FINAL_CONFIG)" ]; then \
	  $(CONFIG_PYTHON) $(SCRIPT_DIR)/config_web_final.py \
	    --write-default-pipeline-config $(FINAL_CONFIG); \
	fi
	$(CONFIG_PYTHON) $(SCRIPT_DIR)/run_hadros_final_pipeline.py $(FINAL_CONFIG)

build-plot-dashboard:
	$(CONFIG_PYTHON) $(SCRIPT_DIR)/build_run_plot_dashboard.py \
	  --run-name $(RUN_NAME)


# =========================================================
# Cleaning
# =========================================================

clean-build:
	rm -rf $(BUILD_DIR)
	rm -f compute_kerr_geodesics dump_kerr_camera_rays

clean-output:
	rm -rf $(OUTPUT_DIR)/particles/*
	rm -rf $(OUTPUT_DIR)/rays/*

clean: clean-build clean-output


# =========================================================
# Automatic dependency tracking
# =========================================================

-include $(BUILD_DIR)/*.d
