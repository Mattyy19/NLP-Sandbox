import tensorflow as tf
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_writer
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from transformers import BertTokenizer

# Paths to files
MODEL_PATH = "minilm_L12_all_lang_ft_v2.11_fp16.tflite"
OUTPUT_PATH = "minilm_L12_all_lang_ft_v2.11_fp16_wmetadata.tflite"
VOCAB_PATH = "tokenizer/vocab.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
for tensor in interpreter.get_input_details():
    print(tensor['name'])

# Load model buffer
print(f"Loading model from: {MODEL_PATH}")
model_buffer = writer_utils.load_file(MODEL_PATH, mode="rb")

# Get input and output tensor information from the model
input_names = writer_utils.get_input_tensor_names(model_buffer)
output_names = writer_utils.get_output_tensor_names(model_buffer)
input_types = writer_utils.get_input_tensor_types(model_buffer)
output_types = writer_utils.get_output_tensor_types(model_buffer)

print(f"Input tensors: {input_names}")
print(f"Output tensors: {output_names}")

input_md = []
for i, (name, tensor_type) in enumerate(zip(input_names, input_types)):
    # Example: Create text input metadata with tokenizer
    tokenizer_md = metadata_info.RegexTokenizerMd(
        delim_regex_pattern=r"\s+",  # Split on whitespace
        vocab_file_path=VOCAB_PATH
    )

    input_tensor_md = metadata_info.InputTextTensorMd(
        name=f"Input {i + 1}",
        description=f"Input tensor: {name}",
        tokenizer_md=tokenizer_md
    )
    input_tensor_md.tensor_name = name  # Important: set the tensor name
    input_md.append(input_tensor_md)

associated_files = [VOCAB_PATH]

print("Creating metadata writer...")
writer = metadata_writer.MetadataWriter.create_from_metadata_info(
    model_buffer=model_buffer,
    input_md=input_md,
    associated_files=associated_files
)

# Populate the metadata into the model
print("Populating metadata into model...")
model_with_metadata = writer.populate()

# Save the model with metadata
print(f"Saving model with metadata to: {OUTPUT_PATH}")
writer_utils.save_file(model_with_metadata, OUTPUT_PATH, mode="wb")

# Print the metadata JSON for verification
print("\nMetadata JSON:")
print(writer.get_populated_metadata_json())

print(f"\nSuccess! Model with metadata saved to: {OUTPUT_PATH}")