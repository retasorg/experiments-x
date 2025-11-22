#!/bin/bash
#
# UNSW-NB15 ML Pipeline Runner
# Executes all 4 training scripts in sequence
#

set -e  # Exit on any error

echo "========================================"
echo "UNSW-NB15 ML Pipeline - Docker"
echo "========================================"
echo ""

# Check if training data exists
if [ ! -f "/app/training_and_test_sets/UNSW_NB15_training-set.csv" ]; then
    echo "❌ ERROR: Training data not found!"
    echo "Please ensure UNSW_NB15_training-set.csv is in ./training_and_test_sets/"
    exit 1
fi

if [ ! -f "/app/training_and_test_sets/UNSW_NB15_testing-set.csv" ]; then
    echo "❌ ERROR: Testing data not found!"
    echo "Please ensure UNSW_NB15_testing-set.csv is in ./training_and_test_sets/"
    exit 1
fi

echo "✓ Training data found"
echo ""

# Create output directories if they don't exist
mkdir -p /app/data
mkdir -p /app/models

# Step 1: Preprocess data
echo "========================================"
echo "Step 1/4: Data Preprocessing"
echo "========================================"
python /app/1_preprocess_data_FIXED.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi
echo ""

# Step 2: Train Isolation Forest
echo "========================================"
echo "Step 2/4: Training Isolation Forest"
echo "========================================"
python /app/2_train_isolation_forest.py
if [ $? -ne 0 ]; then
    echo "❌ Isolation Forest training failed!"
    exit 1
fi
echo ""

# Step 3: Train XGBoost
echo "========================================"
echo "Step 3/4: Training XGBoost"
echo "========================================"
python /app/3_train_xgboost.py
if [ $? -ne 0 ]; then
    echo "❌ XGBoost training failed!"
    exit 1
fi
echo ""

# Step 4: Export to ONNX
echo "========================================"
echo "Step 4/4: Exporting to ONNX"
echo "========================================"
python /app/4_export_to_onnx.py
if [ $? -ne 0 ]; then
    echo "⚠️  ONNX export had errors (this is expected for Isolation Forest)"
    echo "   Check if XGBoost export succeeded"
fi
echo ""

# Summary
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  Data:   /app/data/"
echo "  Models: /app/models/"
echo ""
echo "Check your host ./data and ./models directories"
echo ""
