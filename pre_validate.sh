#!/usr/bin/env bash
echo "Starting Pre-Validation..."

echo "[1/3] Checking OpenEnv files..."
if [ -f "openenv.yaml" ]; then echo "  ✓ openenv.yaml found"; else echo "  ✗ openenv.yaml missing"; exit 1; fi

echo "[2/3] Validating OpenEnv Spec..."
openenv validate

echo "[3/3] Checking Inference Script format..."
if [ -f "inference.py" ]; then echo "  ✓ inference.py found"; else echo "  ✗ inference.py missing"; exit 1; fi

if [ -f "train_trl.py" ]; then echo "  ✓ train_trl.py found"; else echo "  ✗ train_trl.py missing"; exit 1; fi

echo "========================================"
echo "  Ready for Submission!"
echo "========================================"
