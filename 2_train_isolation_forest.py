"""
Isolation Forest Training Script
Trains anomaly detection model on normal traffic only
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time

print("=" * 50)
print("Isolation Forest Training")
print("=" * 50)

# Load preprocessed data
print("\n1. Loading preprocessed data...")
X_normal = np.load('./data/X_normal.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')

print(f"✓ Normal training samples: {len(X_normal)}")
print(f"✓ Test samples: {len(X_test)}")

# Train Isolation Forest
print("\n2. Training Isolation Forest...")
print("Configuration:")
print("  - n_estimators: 100")
print("  - contamination: 0.0 (training on 100% normal data)")
print("  - max_samples: auto")

start_time = time.time()

model = IsolationForest(
    n_estimators=100,
    contamination=0.0,  # Training data has 0% anomalies (novelty detection)
    max_samples='auto',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

model.fit(X_normal)  # Train ONLY on normal traffic

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# Evaluate on test set
print("\n3. Evaluating model...")
# Isolation Forest returns -1 for anomalies, 1 for normal
predictions = model.predict(X_test)

# Convert to binary: -1 (anomaly) -> 1 (attack), 1 (normal) -> 0 (normal)
y_pred = np.where(predictions == -1, 1, 0)

# Calculate metrics
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

# Calculate anomaly scores
print("\n4. Calculating anomaly scores...")
anomaly_scores = model.score_samples(X_test)
print(f"✓ Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
print(f"✓ Mean score: {anomaly_scores.mean():.3f}")

# Save model
print("\n5. Saving model...")
with open('./models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to ./models/isolation_forest.pkl")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"Model: Isolation Forest")
print(f"Training time: {training_time:.2f}s")
print(f"Ready for ONNX export!")
