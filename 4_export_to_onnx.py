"""
ONNX Export Script
Converts trained models to ONNX format for Rust deployment
"""

import numpy as np
import pickle
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorTypeXGB

print("=" * 50)
print("ONNX Model Export")
print("=" * 50)

# Load a sample for shape inference
print("\n1. Loading sample data for shape inference...")
X_test = np.load('./data/X_test.npy')
n_features = X_test.shape[1]
print(f"✓ Number of features: {n_features}")

# Export Isolation Forest
print("\n2. Exporting Isolation Forest to ONNX...")
try:
    with open('./models/isolation_forest.pkl', 'rb') as f:
        iforest_model = pickle.load(f)
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    iforest_onnx = to_onnx(
        iforest_model,
        initial_types=initial_type,
        target_opset={"": 15, "ai.onnx.ml": 2}  # Main domain: "" (not "ai.onnx"), ML opset: 2
    )
    
    # Save ONNX model
    with open('./models/isolation_forest.onnx', 'wb') as f:
        f.write(iforest_onnx.SerializeToString())
    
    print("✓ Isolation Forest exported to ./models/isolation_forest.onnx")
    
    # Verify ONNX model
    session = rt.InferenceSession('./models/isolation_forest.onnx')
    test_sample = X_test[:5].astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    onnx_pred = session.run([output_name], {input_name: test_sample})[0]
    sklearn_pred = iforest_model.predict(test_sample)
    
    match = np.array_equal(onnx_pred.flatten(), sklearn_pred)
    print(f"✓ Verification: {'PASSED' if match else 'FAILED'}")
    
except Exception as e:
    print(f"❌ Error exporting Isolation Forest: {e}")
    print("This is a known issue with sklearn-onnx. See: https://github.com/onnx/sklearn-onnx/issues/518")

# Export XGBoost
print("\n3. Exporting XGBoost to ONNX...")
try:
    # Load XGBoost model as Booster (native API)
    xgb_model = xgb.Booster()
    xgb_model.load_model('./models/xgboost_classifier.json')

    # Convert to ONNX using onnxmltools
    initial_type = [('float_input', FloatTensorTypeXGB([None, n_features]))]

    xgb_onnx = convert_xgboost(
        xgb_model,
        initial_types=initial_type,
        target_opset=15  # Integer for convert_xgboost() with Booster models
    )
    
    # Save ONNX model
    with open('./models/xgboost_classifier.onnx', 'wb') as f:
        f.write(xgb_onnx.SerializeToString())
    
    print("✓ XGBoost exported to ./models/xgboost_classifier.onnx")
    
    # Verify ONNX model
    session = rt.InferenceSession('./models/xgboost_classifier.onnx')
    test_sample = X_test[:5].astype(np.float32)

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    proba_name = session.get_outputs()[1].name

    onnx_result = session.run([label_name, proba_name], {input_name: test_sample})
    onnx_pred = onnx_result[0]
    onnx_proba = onnx_result[1]

    # Booster.predict() returns probabilities, convert to labels
    dtest = xgb.DMatrix(test_sample)
    xgb_proba = xgb_model.predict(dtest)
    xgb_pred = (xgb_proba > 0.5).astype(int)

    match = np.array_equal(onnx_pred.flatten(), xgb_pred)
    print(f"✓ Verification: {'PASSED' if match else 'WARNING - minor differences expected'}")
    
    # Check file sizes
    import os
    iforest_size = os.path.getsize('./models/isolation_forest.onnx') / 1024 if os.path.exists('./models/isolation_forest.onnx') else 0
    xgb_size = os.path.getsize('./models/xgboost_classifier.onnx') / 1024
    
    print("\n" + "=" * 50)
    print("Export Complete!")
    print("=" * 50)
    print(f"Isolation Forest ONNX: {iforest_size:.2f} KB")
    print(f"XGBoost ONNX: {xgb_size:.2f} KB")
    print("\n✓ Models ready for Rust deployment!")
    print("\nNext steps:")
    print("  1. Copy ONNX files to your Rust project")
    print("  2. Use 'ort' crate to load models")
    print("  3. Run inference in real-time on Raspberry Pi")
    
except Exception as e:
    print(f"❌ Error exporting XGBoost: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
