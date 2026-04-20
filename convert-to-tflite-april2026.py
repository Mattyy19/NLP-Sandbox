from transformers import TFAutoModel
import tensorflow as tf
import os
import sys
import traceback

# 1. Define paths
FT_MODEL_PATH = "fine-tune/pm-minilm-L12-v2_wp_all_lang_ft"
TF_SAVED_MODEL_DIR = "saved_model_minilm_L12_ft_april2026"
TFLITE_MODEL_PATH = "minilm_L12_all_lang_ft_float32_april2026.tflite"

def print_error(msg, err):
    print(msg)
    traceback.print_exception(type(err), err, err.__traceback__)
    sys.exit(1)

# 2. Load fine-tuned PyTorch model into TensorFlow
print(f"Loading fine-tuned model from {FT_MODEL_PATH} ...")

if not os.path.exists(FT_MODEL_PATH):
    print(f"Model directory '{FT_MODEL_PATH}' does not exist.")
    sys.exit(1)

try:
    tf_model = TFAutoModel.from_pretrained(FT_MODEL_PATH, from_pt=True)
except Exception as e:
    print_error("Failed to load model.", e)

# 3. Wrap model with mean pooling and replace GELU with tanh approximation
#    (gelu_new uses tanh instead of erfc, which is natively supported by TFLite)
try:
    tf_model.config.hidden_act = "gelu_new"

    class TFEmbeddingModel(tf.Module):
        def __init__(self, tf_model):
            super().__init__()
            self.model = tf_model

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
        def __call__(self, input_ids):
            outputs = self.model(input_ids)
            embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
            embeddings = tf.ensure_shape(embeddings, [None, 384])
            return embeddings

    embedding_model = TFEmbeddingModel(tf_model)
except Exception as e:
    print_error("Failed to wrap TensorFlow model.", e)

# 4. Save as TensorFlow SavedModel
print("Saving TensorFlow SavedModel...")
try:
    tf.saved_model.save(embedding_model, TF_SAVED_MODEL_DIR)
except Exception as e:
    print_error("Failed to save TensorFlow SavedModel.", e)

print(f"SavedModel written to {TF_SAVED_MODEL_DIR}")

# 5. Convert to TFLite (float32, no quantization)
print("Converting SavedModel to TFLite...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_DIR)
    converter.experimental_enable_resource_variables = True
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
except Exception as e:
    print_error("Failed during TFLite conversion.", e)

# 6. Save the .tflite file
try:
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
except Exception as e:
    print_error(f"Failed to write TFLite file to {TFLITE_MODEL_PATH}", e)

print(f"TFLite model written: {TFLITE_MODEL_PATH}")