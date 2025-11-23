"""
XGBoost Classification Training Script
Trains binary classifier for attack detection
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import time
import os
from dotenv import load_dotenv
import wandb
from wandb.integration.xgboost import WandbCallback
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

print("=" * 50)
print("XGBoost Training")
print("=" * 50)

# Initialize wandb run
run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "unsw-nb15-intrusion-detection"),
    name="xgboost-training",
    job_type="train",
    config={
        "model_type": "XGBoost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "objective": "binary:logistic",
        "random_state": 42,
        "eval_metric": "logloss"
    }
)
print("✓ WandB initialized")

# Load preprocessed data
print("\n1. Loading preprocessed data...")
X_train = np.load('./data/X_train.npy')
y_train = np.load('./data/y_train.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Features: {X_train.shape[1]}")

# Convert to DMatrix for native XGBoost API
print("\n2. Preparing data for XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
print("✓ Data converted to DMatrix format")

# Train XGBoost using native API (supports callbacks)
print("\n3. Training XGBoost Classifier...")
print("Configuration:")
print("  - num_boost_round: 100")
print("  - max_depth: 6")
print("  - eta (learning_rate): 0.3")
print("  - objective: binary:logistic")

start_time = time.time()

# XGBoost parameters (native format)
params = {
    'max_depth': 6,
    'eta': 0.3,  # learning_rate
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42,
    'nthread': -1  # Use all CPU cores
}

# Train with WandB callback support
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=True,
    callbacks=[WandbCallback(
        log_model=True,
        log_feature_importance=True,
        importance_type='weight'
    )]
)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# Evaluate on test set
print("\n4. Evaluating model...")
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary predictions

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

# Log comprehensive metrics to wandb
wandb.log({
    "training_time_seconds": training_time,
    "test_accuracy": accuracy,
    "false_positive_rate": fpr,
    "false_negative_rate": fnr,
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
ax.set_title('XGBoost Confusion Matrix')
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close()

# Feature importance
print("\n5. Feature importance (top 10)...")
importance_dict = model.get_score(importance_type='weight')
# Convert to array format (feature index -> importance)
feature_importance = np.zeros(X_train.shape[1])
for feat, score in importance_dict.items():
    feat_idx = int(feat.replace('f', ''))
    feature_importance[feat_idx] = score

top_features = np.argsort(feature_importance)[-10:][::-1]
for i, idx in enumerate(top_features, 1):
    print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")

# Save model
print("\n6. Saving model...")
# Save as pickle (for sklearn compatibility)
with open('./models/xgboost_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to ./models/xgboost_classifier.pkl")

# Save in native XGBoost JSON format (preferred for deployment)
model.save_model('./models/xgboost_classifier.json')
print("✓ Model saved to ./models/xgboost_classifier.json")

# Log model artifacts to wandb (XGBoost formats)
model_artifact = wandb.Artifact(
    name="xgboost-classifier",
    type="model",
    description="Trained XGBoost binary classifier"
)
model_artifact.add_file("./models/xgboost_classifier.pkl")
model_artifact.add_file("./models/xgboost_classifier.json")
run.log_artifact(model_artifact)
print("✓ Model artifacts logged to WandB")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"Model: XGBoost Binary Classifier")
print(f"Training time: {training_time:.2f}s")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Ready for ONNX export!")

# Finish wandb run
wandb.finish()
print("✓ WandB run finished")
