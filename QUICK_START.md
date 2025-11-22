# 🚀 Quick Start Guide (CORRECTED)

## ⚠️ IMPORTANT: Use the FIXED Version

The original `1_preprocess_data.py` has bugs. **Use the FIXED version instead!**

## 📦 Files You Have

### ✅ Use These Files (In Order):
1. **`1_preprocess_data_FIXED.py`** ← Use this one!
2. **`2_train_isolation_forest.py`** ← Original is OK
3. **`3_train_xgboost.py`** ← Original is OK  
4. **`4_export_to_onnx.py`** ← Original is OK

### ❌ Don't Use:
- `1_preprocess_data.py` ← Has CSV header bug

## 🎯 Installation & Setup

```bash
# 1. Create environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
mkdir -p data models

# 4. Download UNSW-NB15 dataset
# Get from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Files needed:
#   - UNSW_NB15_training-set.csv
#   - UNSW_NB15_testing-set.csv
# Place in ./data/
```

## ▶️ Run Training

```bash
# Step 1: Preprocess (FIXED VERSION)
python 1_preprocess_data_FIXED.py

# Step 2: Train Isolation Forest
python 2_train_isolation_forest.py

# Step 3: Train XGBoost
python 3_train_xgboost.py

# Step 4: Export to ONNX
python 4_export_to_onnx.py
```

## 🔍 Verify It Worked

```bash
# Check preprocessed data
ls -lh ./data/*.npy

# Check trained models
ls -lh ./models/*.pkl ./models/*.json

# Check ONNX models (for Rust)
ls -lh ./models/*.onnx
```

Expected output:
```
./data/X_train.npy        - ~60 MB
./data/X_test.npy         - ~30 MB
./data/X_normal.npy       - ~50 MB
./models/isolation_forest.pkl - ~5 MB
./models/xgboost_classifier.pkl - ~10 MB
./models/xgboost_classifier.onnx - ~10 MB ← For Rust!
```

## 🦀 Use in Rust

```rust
// Cargo.toml
[dependencies]
ort = "2.0"
ndarray = "0.15"

// main.rs
use ort::{Environment, SessionBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = Environment::builder().build()?;
    
    // Load models
    let xgb = SessionBuilder::new(&env)?
        .with_model_from_file("xgboost_classifier.onnx")?;
    
    // Run inference...
    Ok(())
}
```

## ❓ Troubleshooting

### CSV header error
```
ValueError: Length mismatch: Expected 49 columns
```
**Fix:** Make sure you use `1_preprocess_data_FIXED.py`

### Memory error
```
MemoryError: Unable to allocate array
```
**Fix:** Process in batches or use smaller sample:
```python
# In preprocessing script, add:
train_df = train_df.sample(n=50000)  # Use subset
```

### ONNX export fails
```
Error converting Isolation Forest
```
**Fix:** This is a known issue. XGBoost export should work. If needed:
```bash
# Update packages
pip install --upgrade onnx onnxruntime skl2onnx onnxmltools
```

## 📊 Expected Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Isolation Forest | 85-95% | Trained on normal traffic only |
| XGBoost | 95-99% | Binary classification |

## 🎓 What's Different from Plan

**Original Plan:** 3 models (FastText + Isolation Forest + XGBoost)

**Actual Implementation:** 2 models (Isolation Forest + XGBoost)

**Why:** FastText is designed for text data. UNSW-NB15 is mostly numerical network flow features, so FastText doesn't apply here. This is standard practice in the research.

## ✅ Final Checklist

- [ ] Downloaded UNSW-NB15 dataset
- [ ] Placed CSV files in ./data/
- [ ] Installed requirements.txt
- [ ] Run `1_preprocess_data_FIXED.py` (not the original!)
- [ ] Trained both models successfully
- [ ] Exported to ONNX
- [ ] Have .onnx files ready for Rust

## 🔗 Need Help?

- Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Rust ort crate: https://github.com/pykeio/ort
- Read: `ISSUES_AND_FIXES.md` for technical details

---

**Remember: Use the FIXED preprocessing script!** 🎯
