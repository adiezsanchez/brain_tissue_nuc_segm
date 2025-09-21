#!/usr/bin/env bash
set -euo pipefail

echo "🔎 Checking OpenCL setup inside Pixi env..."

# Where Pixi puts the env
PIXI_ENV="${PIXI_PROJECT_ROOT:-$(pwd)}/.pixi/envs/default"

# Ensure libOpenCL.so is linked
if [ ! -e "$PIXI_ENV/lib/libOpenCL.so" ]; then
  echo "➡ Linking libOpenCL.so into Pixi env..."
  ln -sf /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 "$PIXI_ENV/lib/libOpenCL.so"
else
  echo "✅ libOpenCL.so already linked."
fi

# Ensure ICD vendor dir exists
mkdir -p "$PIXI_ENV/etc/OpenCL/vendors"

# Link NVIDIA ICD file
if [ ! -e "$PIXI_ENV/etc/OpenCL/vendors/nvidia.icd" ]; then
  echo "➡ Linking nvidia.icd into Pixi env..."
  ln -sf /etc/OpenCL/vendors/nvidia.icd "$PIXI_ENV/etc/OpenCL/vendors/nvidia.icd"
else
  echo "✅ nvidia.icd already linked."
fi

echo "✅ OpenCL setup complete inside Pixi env."



