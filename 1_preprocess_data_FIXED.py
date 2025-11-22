"""
UNSW-NB15 Data Preprocessing Script (FIXED VERSION)
Simple preprocessing for DMZ Gateway proof-of-concept

FIXES:
- Correct file paths (./training_and_test_sets/)
- Correct feature count (45 columns from CSV headers)
- Proper handling of 'id' column
- One-hot encoding for categorical features (proto, service, state)
- Better missing value handling
- Correct label interpretation (0=normal, 1=attack)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

print("=" * 50)
print("UNSW-NB15 Data Preprocessing (FIXED)")
print("=" * 50)

# UNSW-NB15 Dataset Information:
# - The CSV files include headers with 45 columns
# - Features include network flow statistics, protocol info, and behavioral patterns
# - Categorical features (proto, service, state) are one-hot encoded
# - Final feature count: ~100-200 after one-hot encoding
# - Target: 'label' (0=Normal, 1=Attack) and 'attack_cat' (attack category)
# - Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

print("\n1. Loading dataset...")
print("Files needed: UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
print("Place them in ./training_and_test_sets/ directory")

# Create data and models directories if they don't exist
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)

try:
    # Load CSV files (files include headers)
    train_df = pd.read_csv('./training_and_test_sets/UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('./training_and_test_sets/UNSW_NB15_testing-set.csv')

    print(f"✓ Training set: {train_df.shape}")
    print(f"✓ Testing set: {test_df.shape}")
    print(f"✓ Columns: {len(train_df.columns)}")

    # Validate required columns exist
    required_cols = ['label', 'proto', 'service', 'state', 'attack_cat', 'id']
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        exit(1)
    print("✓ All required columns present")

except FileNotFoundError:
    print("❌ Error: Dataset files not found!")
    print("\nDownload from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("You need:")
    print("  - UNSW_NB15_training-set.csv")
    print("  - UNSW_NB15_testing-set.csv")
    exit(1)

print("\n2. Preprocessing data...")

# Drop columns that aren't useful for ML
# - 'id': row identifier (not a feature)
# - 'attack_cat': attack category name (we use 'label' instead for binary classification)
cols_to_drop = ['id', 'attack_cat']
train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

print(f"✓ Dropped {len(cols_to_drop)} non-feature columns")

# Handle missing values BEFORE splitting features/labels
print("\n3. Handling missing values...")
# Check for columns with nulls
null_cols = train_df.columns[train_df.isnull().any()].tolist()
if null_cols:
    print(f"⚠ Found nulls in: {null_cols}")
    # Fill numeric columns with 0, categorical with mode
    for col in null_cols:
        if train_df[col].dtype in ['int64', 'float64']:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)
        else:
            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 'unknown'
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
    print("✓ Missing values handled")
else:
    print("✓ No missing values found")

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"\n✓ Features: {X_train.shape[1]}")
print(f"✓ Samples - Train: {len(X_train)}, Test: {len(X_test)}")

# IMPORTANT: Label 0 = Normal, Label 1 = Attack
normal_count = (y_train == 0).sum()
attack_count = (y_train == 1).sum()
print(f"✓ Label distribution (train):")
print(f"    Normal (0): {normal_count} ({normal_count/len(y_train)*100:.1f}%)")
print(f"    Attack (1): {attack_count} ({attack_count/len(y_train)*100:.1f}%)")

# One-hot encode categorical features
print("\n4. One-hot encoding categorical features...")
# Main categorical columns in UNSW-NB15
categorical_cols = ['proto', 'service', 'state']

# Perform one-hot encoding
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, prefix=categorical_cols)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, prefix=categorical_cols)

print(f"✓ One-hot encoded {len(categorical_cols)} categorical features")
print(f"✓ Features before encoding: {len(X_train.columns)}")
print(f"✓ Features after encoding: {len(X_train_encoded.columns)}")

# Align columns between train and test (handle unseen categories)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# Sort columns for consistency
X_train_encoded = X_train_encoded.sort_index(axis=1)
X_test_encoded = X_test_encoded.sort_index(axis=1)

print(f"✓ Aligned train/test columns: {len(X_train_encoded.columns)} features")

# Update references
X_train = X_train_encoded
X_test = X_test_encoded

# Verify all columns are numeric (should be after one-hot encoding)
print("\n5. Verifying data types...")
non_numeric = [col for col in X_train.columns if X_train[col].dtype == 'object']
if non_numeric:
    print(f"⚠ Converting {len(non_numeric)} non-numeric columns")
    for col in non_numeric:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
else:
    print("✓ All features are numeric")

# Normalize features
print("\n6. Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features normalized using StandardScaler")

# Extract NORMAL traffic for Isolation Forest (label=0 is normal)
print("\n7. Extracting normal traffic for Isolation Forest...")
X_normal = X_train_scaled[y_train == 0]  # Label 0 = Normal traffic
print(f"✓ Normal traffic samples: {len(X_normal)} ({len(X_normal)/len(X_train_scaled)*100:.1f}%)")
print(f"✓ Attack samples: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

# Save preprocessed data
print("\n8. Saving preprocessed data...")
np.save('./data/X_train.npy', X_train_scaled)
np.save('./data/y_train.npy', y_train.values)
np.save('./data/X_test.npy', X_test_scaled)
np.save('./data/y_test.npy', y_test.values)
np.save('./data/X_normal.npy', X_normal)

# Save preprocessing artifacts
with open('./data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved scaler")

# Save feature names for reference
with open('./data/feature_names.txt', 'w') as f:
    f.write('\n'.join(X_train.columns.tolist()))

print("✓ All files saved to ./data/")

print("\n" + "=" * 50)
print("Preprocessing Complete!")
print("=" * 50)
print(f"Training samples: {len(X_train_scaled):,}")
print(f"Testing samples: {len(X_test_scaled):,}")
print(f"Features: {X_train_scaled.shape[1]} (45 original - 2 dropped + one-hot encoding)")
print(f"  - Categorical features one-hot encoded: proto, service, state")
print(f"Normal traffic: {len(X_normal):,} (for Isolation Forest)")
print(f"Attack traffic: {(y_train == 1).sum():,} (for XGBoost)")
print("\n✅ Ready for model training!")
print("Next: Run 2_train_isolation_forest.py and 3_train_xgboost.py")
