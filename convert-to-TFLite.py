# convert_to_tflite.py
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import os

# 1. Define paths
FT_MODEL_PATH = "minilm-L6-v2_wikipedia100_ft"
TF_SAVED_MODEL_DIR = "saved_model_minilm_ft"
TFLITE_MODEL_PATH = "minilm_ft_fp16.tflite"

# 2. Load fine-tuned model from PyTorch checkpoint
print(f"Loading fine-tuned model from {FT_MODEL_PATH} ...")
tf_model = TFAutoModel.from_pretrained(FT_MODEL_PATH, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 3. Wrap TF model to output mean-pooled sentence embeddings
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

# 4. Save as TensorFlow SavedModel
print("Saving TensorFlow SavedModel...")
tf.saved_model.save(embedding_model, TF_SAVED_MODEL_DIR)
print(f"Saved TensorFlow model to {TF_SAVED_MODEL_DIR}")

# 5. Convert to TensorFlow Lite
print("Converting SavedModel to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the converter to use float16 for reduced model size and faster inference
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# 6. Save the .tflite model
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved as {TFLITE_MODEL_PATH}")
