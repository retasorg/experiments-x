# UNSW-NB15 ML Pipeline - Docker Setup

Automated ML training pipeline for UNSW-NB15 network intrusion detection with GPU support.

## Prerequisites

1. **Docker** (20.10+)
2. **Docker Compose** (v2.0+)
3. **NVIDIA Docker Runtime** (for GPU support)
   ```bash
   # Install nvidia-docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

4. **UNSW-NB15 Dataset** in `./training_and_test_sets/`
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

## Quick Start

### 1. Verify GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Run the Pipeline
```bash
docker compose up
```

That's it! The pipeline will:
1. Preprocess data (one-hot encoding, normalization)
2. Train Isolation Forest
3. Train XGBoost
4. Export models to ONNX format

### 3. Check Output
```bash
ls -lh ./data/     # Preprocessed data
ls -lh ./models/   # Trained models (.pkl, .json, .onnx)
```

## Directory Structure

```
.
├── Dockerfile                          # Docker image definition
├── docker-compose.yml                  # Docker Compose config
├── requirements.txt                    # Python dependencies
├── run_pipeline.sh                     # Pipeline runner script
├── 1_preprocess_data_FIXED.py
├── 2_train_isolation_forest.py
├── 3_train_xgboost.py
├── 4_export_to_onnx.py
├── training_and_test_sets/             # Input data (you provide)
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── data/                               # Output: preprocessed data
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   ├── y_test.npy
│   ├── X_normal.npy
│   ├── scaler.pkl
│   └── feature_names.txt
└── models/                             # Output: trained models
    ├── isolation_forest.pkl
    ├── xgboost_classifier.pkl
    ├── xgboost_classifier.json
    ├── isolation_forest.onnx
    └── xgboost_classifier.onnx
```

## Manual Steps

### Build Only
```bash
docker compose build
```

### Run Interactively
```bash
docker compose run --rm ml-pipeline bash
# Inside container:
python 1_preprocess_data_FIXED.py
python 2_train_isolation_forest.py
python 3_train_xgboost.py
python 4_export_to_onnx.py
```

### View Logs
```bash
docker compose logs -f
```

### Clean Up
```bash
docker compose down
docker rmi unsw-nb15-ml-pipeline:latest
```

## Expected Output

### Console Output
```
========================================
UNSW-NB15 ML Pipeline - Docker
========================================

✓ Training data found

========================================
Step 1/4: Data Preprocessing
========================================
==================================================
UNSW-NB15 Data Preprocessing (FIXED)
==================================================
...
✓ Training set: (175341, 45)
✓ Testing set: (82332, 45)
✓ One-hot encoded 3 categorical features
...

========================================
Step 2/4: Training Isolation Forest
========================================
...
✓ Test Accuracy: 0.XXXX

========================================
Step 3/4: Training XGBoost
========================================
...
✓ Test accuracy: 0.87XX

========================================
Step 4/4: Exporting to ONNX
========================================
...
✓ XGBoost exported to ./models/xgboost_classifier.onnx

========================================
Pipeline Complete!
========================================
```

### Performance Expectations
- **Preprocessing:** ~30-60 seconds
- **Isolation Forest:** ~2-5 minutes
- **XGBoost:** ~5-15 minutes (with GPU)
- **ONNX Export:** ~10-30 seconds
- **Total Time:** ~10-20 minutes

## Troubleshooting

### GPU Not Detected
```bash
# Check nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-docker runtime
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Dataset Not Found
```
❌ ERROR: Training data not found!
```
**Solution:** Ensure CSV files are in `./training_and_test_sets/`:
```bash
ls -lh ./training_and_test_sets/
# Should show:
# UNSW_NB15_training-set.csv
# UNSW_NB15_testing-set.csv
```

### ONNX Export Fails (Isolation Forest)
```
❌ Error exporting Isolation Forest: ...
This is a known issue with sklearn-onnx.
```
**Solution:** This is expected! IsolationForest ONNX export is fragile. Check if XGBoost export succeeded:
```bash
ls -lh ./models/xgboost_classifier.onnx
```
XGBoost export is more reliable and sufficient for the PoC.

### Out of Memory
If you get OOM errors, reduce XGBoost parameters in `3_train_xgboost.py`:
```python
n_estimators=50  # Reduce from 100
max_depth=4      # Reduce from 6
```

### Permission Denied (Output Files)
```bash
sudo chown -R $USER:$USER ./data ./models
```

## CPU-Only Mode

If you don't have GPU or nvidia-docker:

1. Edit `requirements.txt`:
   ```
   # Change:
   onnxruntime-gpu>=1.16.0
   # To:
   onnxruntime>=1.16.0
   ```

2. Edit `docker-compose.yml`:
   ```yaml
   # Comment out the deploy section:
   # deploy:
   #   resources:
   #     reservations:
   #       devices:
   #         - driver: nvidia
   #           count: 1
   #           capabilities: [gpu]
   ```

3. Use CPU base image in `Dockerfile`:
   ```dockerfile
   # Change FROM line:
   FROM python:3.10-slim
   ```

## Next Steps

After successful training:
1. Check accuracy in console output
2. Copy ONNX models to Raspberry Pi
3. Deploy using Rust + `ort` crate
4. Test real-time inference

## Support

For issues:
- Check Docker logs: `docker compose logs`
- Check GPU: `nvidia-smi`
- Verify data: `ls -lh ./training_and_test_sets/`
