# convert_to_tflite.py
from transformers import AutoTokenizer, TFAutoModel, AutoModel
import tensorflow as tf
import os
import sys
import traceback

# 1. Define paths
FT_MODEL_PATH = "fine-tune/pm-minilm-L12-v2_wp_all_lang_ft"
TF_SAVED_MODEL_DIR = "saved_model_minilm_L12_ft"
TFLITE_MODEL_PATH = "minilm_L12_all_lang_ft_fp16.tflite"

def print_error(msg, err):
    print(msg)
    traceback.print_exception(type(err), err, err.__traceback__)
    sys.exit(1)

# 2. Load fine-tuned model
print(f"Loading fine-tuned model from {FT_MODEL_PATH} ...")

if not os.path.exists(FT_MODEL_PATH):
    print(f"Model directory '{FT_MODEL_PATH}' does not exist.")
    sys.exit(1)

try:
    tf_model = TFAutoModel.from_pretrained(FT_MODEL_PATH, from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    tokenizer.save_pretrained("tokenizer/")
except Exception as e:
    print_error("Failed to load model or tokenizer.", e)

# 3. Wrap TF model to output mean-pooled sentence embeddings
try:
    class TFEmbeddingModel(tf.Module):
        def __init__(self, tf_model):
            super().__init__()
            self.model = tf_model

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
        def __call__(self, input_ids):
            outputs = self.model(input_ids)
            embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
            return embeddings

    embedding_model = TFEmbeddingModel(tf_model)
except Exception as e:
    print_error("Failed to wrap TensorFlow model.")

# 4. Save as TensorFlow SavedModel
print("Saving TensorFlow SavedModel...")
try:
    tf.saved_model.save(embedding_model, TF_SAVED_MODEL_DIR)
except Exception as e:
    print_error("Failed to save TensorFlow SavedModel.", e)

print(f"Saved TensorFlow model to {TF_SAVED_MODEL_DIR}")

# 5. Convert to TensorFlow Lite
print("Converting SavedModel to TensorFlow Lite format...")

try:
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set the converter to use float16 for reduced model size and faster inference
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
except Exception as e:
    print_error("Failed during TFLite conversion.")

# 6. Save the .tflite model
try:
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
except Exception as e:
    print_error(f"Faied to write TFLite file to {TFLITE_MODEL_PATH}", e)

print(f"TFLite model saved as {TFLITE_MODEL_PATH}")
