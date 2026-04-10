import tensorflow as tf
import numpy as np

SAVED_MODEL_DIR = "saved_model_minilm_L12_ft_april2026"
TFLITE_OUT_AUDIT = "minilm_L12_all_lang_ft_float32_april2026.tflite"

# ---------------------------------------------------------------------------
# STEP 1: Inspect the SavedModel graph
# Check what signatures and outputs are available, so we can verify that
# mean-pooling and normalization are captured (not just raw token embeddings).
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Inspecting SavedModel signatures")
print("=" * 60)

loaded = tf.saved_model.load(SAVED_MODEL_DIR)
print("Signatures found:", list(loaded.signatures.keys()))

infer = loaded.signatures["serving_default"]
print("\nInputs:")
for k, v in infer.structured_input_signature[1].items():
    print(f"  {k}: {v}")
print("\nOutputs:")
for k, v in infer.structured_outputs.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# ---------------------------------------------------------------------------
# STEP 2: Loading float32 TFLite model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Loading float32 TFLite model")
print("=" * 60)

SEQ_LEN = 128
interpreter = tf.lite.Interpreter(model_path=TFLITE_OUT_AUDIT)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
interpreter.resize_tensor_input(input_index, [1, SEQ_LEN])
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite input tensors:")
for d in input_details:
    print(f"  [{d['index']}] {d['name']:40s} shape={d['shape']} dtype={d['dtype'].__name__}")

print("\nTFLite output tensors:")
for d in output_details:
    print(f"  [{d['index']}] {d['name']:40s} shape={d['shape']} dtype={d['dtype'].__name__}")

# ---------------------------------------------------------------------------
# STEP 3: Compare SavedModel vs float32 TFLite embeddings on sample inputs
# Both should produce near-identical outputs (cosine sim ~1.0, MSE ~0.0).
# If they don't, the SavedModel graph is missing pooling/normalization.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Comparing SavedModel vs float32 TFLite embeddings")
print("=" * 60)

dummy_input_ids      = np.ones((1, SEQ_LEN), dtype=np.int32)
dummy_attention_mask = np.ones((1, SEQ_LEN), dtype=np.int32)
dummy_token_type_ids = np.zeros((1, SEQ_LEN), dtype=np.int32)

# SavedModel inference
sm_outputs = infer(
    input_ids=tf.constant(dummy_input_ids),
)
sm_key = list(sm_outputs.keys())[0]
sm_embeddings = sm_outputs[sm_key].numpy()
print(f"SavedModel output key: '{sm_key}', shape: {sm_embeddings.shape}")

# TFLite inference
for detail in input_details:
    name = detail["name"].lower()
    if "input_ids" in name:
        interpreter.set_tensor(detail["index"], dummy_input_ids)
    elif "attention_mask" in name:
        interpreter.set_tensor(detail["index"], dummy_attention_mask)
    elif "token_type_ids" in name:
        interpreter.set_tensor(detail["index"], dummy_token_type_ids)

interpreter.invoke()
tflite_embeddings = interpreter.get_tensor(output_details[0]["index"])
print(f"TFLite output shape: {tflite_embeddings.shape}")

# Comparison metrics
mse = np.mean((sm_embeddings - tflite_embeddings) ** 2)

def cosine_sim(a, b):
    dot = np.sum(a * b, axis=-1)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return dot / (norm + 1e-9)

cos_sims = cosine_sim(sm_embeddings, tflite_embeddings)

print("\n--- Accuracy Audit Results ---")
print(f"MSE (SavedModel vs TFLite float32): {mse:.6f}")
print(f"Cosine similarity per sample:       {cos_sims}")
print()

if mse < 1e-4 and np.all(cos_sims > 0.999):
    print("PASS: Float32 TFLite matches SavedModel. The Erfc/GELU fix worked.")
    print("      Embeddings are faithful to the original model.")
else:
    print("FAIL: TFLite output still diverges from SavedModel.")
    print(f"      Output shapes - SavedModel: {sm_embeddings.shape}, TFLite: {tflite_embeddings.shape}")
