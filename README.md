# UNSW-NB15 ML Training Scripts

Simple Python scripts to train Isolation Forest and XGBoost models on UNSW-NB15 dataset for DMZ Gateway proof-of-concept.

## 🎯 Goal

Train two ML models for malicious traffic detection:
1. **Isolation Forest** - Anomaly detection (unsupervised)
2. **XGBoost** - Binary classification (supervised)

Export both to **ONNX format** for deployment in Rust on Raspberry Pi.

## 📁 Project Structure

```
.
├── 1_preprocess_data.py          # Data loading and preprocessing
├── 2_train_isolation_forest.py   # Train anomaly detection model
├── 3_train_xgboost.py            # Train classification model
├── 4_export_to_onnx.py           # Export models to ONNX
├── requirements.txt               # Python dependencies
├── data/                          # Dataset directory (create this)
│   └── [UNSW-NB15 CSV files here]
└── models/                        # Trained models (created automatically)
    ├── *.pkl                      # Pickle models
    ├── *.json                     # XGBoost native format
    └── *.onnx                     # ONNX models for Rust
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download UNSW-NB15 from: https://research.unsw.edu.au/projects/unsw-nb15-dataset

You need these files:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Place them in `./data/` directory:

```bash
mkdir -p data models
# Copy CSV files to ./data/
```

### 3. Run Training Pipeline

Execute scripts in order:

```bash
# Step 1: Preprocess data
python 1_preprocess_data.py

# Step 2: Train Isolation Forest
python 2_train_isolation_forest.py

# Step 3: Train XGBoost
python 3_train_xgboost.py

# Step 4: Export to ONNX
python 4_export_to_onnx.py
```

### 4. Output Files

After training, you'll have:

```
models/
├── isolation_forest.pkl          # Scikit-learn model
├── isolation_forest.onnx         # ONNX model for Rust ✓
├── xgboost_classifier.pkl        # XGBoost pickle
├── xgboost_classifier.json       # XGBoost native
└── xgboost_classifier.onnx       # ONNX model for Rust ✓
```

The `.onnx` files are ready for Rust deployment!

## 📊 Expected Results

Based on UNSW-NB15 research papers:

| Model             | Expected Accuracy | Notes                          |
|-------------------|-------------------|--------------------------------|
| Isolation Forest  | 85-95%            | Unsupervised anomaly detection |
| XGBoost           | 95-99%            | Supervised classification      |

**Note**: These are baseline models for hardware demonstration. Production deployments should train on actual network traffic.

## 🔧 Customization

### Isolation Forest Parameters

Edit `2_train_isolation_forest.py`:

```python
model = IsolationForest(
    n_estimators=100,      # Number of trees (increase for better accuracy)
    contamination=0.1,     # Expected % of anomalies (tune this)
    max_samples='auto',    # Samples per tree
    random_state=42
)
```

### XGBoost Parameters

Edit `3_train_xgboost.py`:

```python
model = xgb.XGBClassifier(
    n_estimators=100,      # Number of boosting rounds
    max_depth=6,           # Tree depth (increase carefully)
    learning_rate=0.3,     # Step size (lower = slower but more accurate)
    random_state=42
)
```

## 🦀 Rust Integration

### Load ONNX in Rust

Add to `Cargo.toml`:

```toml
[dependencies]
ort = "2.0"
ndarray = "0.15"
```

Example Rust code:

```rust
use ort::{Environment, SessionBuilder, Value};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load models
    let env = Environment::builder().build()?;
    
    let iforest = SessionBuilder::new(&env)?
        .with_model_from_file("isolation_forest.onnx")?;
    
    let xgboost = SessionBuilder::new(&env)?
        .with_model_from_file("xgboost_classifier.onnx")?;
    
    // Prepare input (example: 1 sample with 43 features)
    let input = Array2::<f32>::zeros((1, 43));
    let input_tensor = Value::from_array(input)?;
    
    // Run inference
    let iforest_outputs = iforest.run(vec![input_tensor.clone()])?;
    let xgboost_outputs = xgboost.run(vec![input_tensor])?;
    
    // Process results...
    println!("Inference complete!");
    
    Ok(())
}
```

## ⚠️ Known Issues

### Isolation Forest ONNX Export

<cite index="48-1">The Isolation Forest ONNX conversion may have compatibility issues with some sklearn versions</cite>. If export fails:

1. Try downgrading: `pip install scikit-learn==1.3.0`
2. Or use the pickle model directly in a Python service
3. The XGBoost model should export successfully

### Large Model Size

XGBoost ONNX models can be large (10-50 MB). For Raspberry Pi:
- Reduce `n_estimators` (e.g., 50 instead of 100)
- Reduce `max_depth` (e.g., 4 instead of 6)

## 📝 Citation

If using UNSW-NB15 dataset, please cite:

```
Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for 
network intrusion detection systems (UNSW-NB15 network data set)." 
Military Communications and Information Systems Conference (MilCIS), 2015.
```

## 🎯 Remember

**This is a proof-of-concept!**

- Models are trained on UNSW-NB15 (2015 synthetic data)
- Real deployments need training on actual network traffic
- Customers should train their own models for their specific environment
- Goal: Demonstrate hardware capability, not production-ready models

## 🆘 Troubleshooting

### Dataset not found
```bash
# Make sure CSV files are in ./data/
ls -la ./data/
```

### Memory errors
```python
# Reduce data size in preprocessing script
train_df = train_df.sample(n=50000)  # Use subset
```

### ONNX conversion fails
```bash
# Update packages
pip install --upgrade onnx onnxruntime skl2onnx onnxmltools
```

## 📚 References

- UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- ONNX Runtime: https://onnxruntime.ai/
- sklearn-onnx: https://onnx.ai/sklearn-onnx/
- Rust ort crate: https://github.com/pykeio/ort

---

**Next Steps**: Deploy these ONNX models in your Rust DMZ Gateway! 🚀
