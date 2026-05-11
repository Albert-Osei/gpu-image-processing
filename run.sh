#!/bin/bash

# ============================================================================
# CUDA Image Processing Pipeline Runner
# ============================================================================
# This script:
#   1. Cleans previous builds/results
#   2. Builds the CUDA project
#   3. Generates synthetic test images
#   4. Runs the GPU image processing pipeline
#   5. Displays summary information
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
# ============================================================================

set -e

echo "============================================================"
echo " CUDA Image Processing Pipeline"
echo "============================================================"

# ----------------------------------------------------------------------------
# Clean previous outputs
# ----------------------------------------------------------------------------
echo ""
echo "[1/4] Cleaning previous build/output files..."
make clean

# ----------------------------------------------------------------------------
# Build CUDA application
# ----------------------------------------------------------------------------
echo ""
echo "[2/4] Building CUDA application..."
make build

# ----------------------------------------------------------------------------
# Generate synthetic test images
# ----------------------------------------------------------------------------
echo ""
echo "[3/4] Generating synthetic images..."
make generate_data

# ----------------------------------------------------------------------------
# Run image processing pipeline
# ----------------------------------------------------------------------------
echo ""
echo "[4/4] Running GPU image processing..."
./bin/image_processing.exe ./data/images ./output ./results.csv