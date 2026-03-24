import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="minilm_L12_all_lang_ft_v2.11_fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input tensors:")
for inp in input_details:
    print(f"Name: {inp['name']}, shape: {inp['shape']}, dtype: {inp['dtype']}")

print("\nOutput tensors:")
for out in output_details:
    print(f"Name: {out['name']}, shape: {out['shape']}, dtype: {out['dtype']}")