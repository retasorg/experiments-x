"""
XGBoost Classification Training Script
Trains binary classifier for attack detection
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import time

print("=" * 50)
print("XGBoost Training")
print("=" * 50)

# Load preprocessed data
print("\n1. Loading preprocessed data...")
X_train = np.load('./data/X_train.npy')
y_train = np.load('./data/y_train.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Features: {X_train.shape[1]}")

# Train XGBoost
print("\n2. Training XGBoost Classifier...")
print("Configuration:")
print("  - n_estimators: 100")
print("  - max_depth: 6")
print("  - learning_rate: 0.3")
print("  - objective: binary:logistic")

start_time = time.time()

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    eval_metric='logloss'
)

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# Evaluate on test set
print("\n3. Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Accuracy: {accuracy:.4f}")

print("\n" + "=" * 50)
print("Classification Report:")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0][0]} (Correctly identified normal)")
print(f"False Positives: {cm[0][1]} (Normal flagged as attack)")
print(f"False Negatives: {cm[1][0]} (Attack missed)")
print(f"True Positives:  {cm[1][1]} (Correctly identified attack)")

# Calculate false positive rate (important for DMZ gateway)
fpr = cm[0][1] / (cm[0][0] + cm[0][1])
fnr = cm[1][0] / (cm[1][0] + cm[1][1])
print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
print(f"False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")

# Feature importance
print("\n4. Feature importance (top 10)...")
feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-10:][::-1]
for i, idx in enumerate(top_features, 1):
    print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")

# Save model
print("\n5. Saving model...")
with open('./models/xgboost_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to ./models/xgboost_classifier.pkl")

# Also save in native XGBoost format for better compatibility
model.save_model('./models/xgboost_classifier.json')
print("✓ Model saved to ./models/xgboost_classifier.json")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"Model: XGBoost Binary Classifier")
print(f"Training time: {training_time:.2f}s")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Ready for ONNX export!")
