"""
Isolation Forest Training Script
Trains anomaly detection model on normal traffic only
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pickle
import time
import os
from dotenv import load_dotenv
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

print("=" * 50)
print("Isolation Forest Training")
print("=" * 50)

# Initialize wandb run
run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "unsw-nb15-intrusion-detection"),
    name="isolation-forest-training",
    job_type="train",
    config={
        "model_type": "IsolationForest",
        "n_estimators": 100,
        "contamination": "auto",
        "max_samples": "auto",
        "random_state": 42
    }
)
print("✓ WandB initialized")

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
print("  - contamination: auto (novelty detection with clean training data)")
print("  - max_samples: auto")

start_time = time.time()

model = IsolationForest(
    n_estimators=100,
    contamination='auto',  # 'auto' for novelty detection (training on clean data)
    max_samples='auto',
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

model.fit(X_normal)  # Train ONLY on normal traffic

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# Log training metrics to wandb
wandb.log({
    "training_time_seconds": training_time,
    "training_samples": len(X_normal)
})

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

# Calculate and log evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = (cm[0][0] + cm[1][1]) / cm.sum()

wandb.log({
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall,
    "test_f1_score": f1,
    "true_negatives": int(cm[0][0]),
    "false_positives": int(cm[0][1]),
    "false_negatives": int(cm[1][0]),
    "true_positives": int(cm[1][1])
})

# Log confusion matrix as image
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Isolation Forest Confusion Matrix')
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close()

# Calculate anomaly scores
print("\n4. Calculating anomaly scores...")
anomaly_scores = model.score_samples(X_test)
print(f"✓ Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
print(f"✓ Mean score: {anomaly_scores.mean():.3f}")

# Log anomaly scores to wandb
wandb.log({
    "anomaly_score_min": float(anomaly_scores.min()),
    "anomaly_score_max": float(anomaly_scores.max()),
    "anomaly_score_mean": float(anomaly_scores.mean())
})

# Save model
print("\n5. Saving model...")
with open('./models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to ./models/isolation_forest.pkl")

# Log model artifact to wandb
model_artifact = wandb.Artifact(
    name="isolation-forest-model",
    type="model",
    description="Trained IsolationForest for anomaly detection"
)
model_artifact.add_file("./models/isolation_forest.pkl")
run.log_artifact(model_artifact)
print("✓ Model artifact logged to WandB")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"Model: Isolation Forest")
print(f"Training time: {training_time:.2f}s")
print(f"Ready for ONNX export!")

# Finish wandb run
wandb.finish()
print("✓ WandB run finished")
