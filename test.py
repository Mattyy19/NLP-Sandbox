import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import psutil
import os
import time
import json

interpreter = tf.lite.Interpreter(model_path="minilm_L12_all_lang_ft_v2.11_fp16_wmetadata.tflite")

input_details = interpreter.get_input_details()

interpreter.resize_tensor_input(input_details[0]['index'], [1, 128])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:")
for inp in input_details:
    print(inp)

print("\nOutput Details:")
for out in output_details:
    print(out)

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\fine-tune\pm-minilm-L12-v2_wp_all_lang_v2.11_ft")

# Gets system process info to track cpu + ram
process = psutil.Process(os.getpid())
cpu_count = psutil.cpu_count(logical=True)

# Testing sample, change in the future
titles = []
sections = []
chunk_ids = []
texts = []
file_path = r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\fine-tune\wp_paragraph.jsonl"

def parse_jsonl():
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoing JSON on line: {e}")
                    continue

for data in parse_jsonl():
    title = data.get("title", "")
    text = data.get("text", "")

    if text:
        titles.append(title)
        texts.append(text)

def preprocess(texts, max_length=128):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    return encoded

def embed_batch(texts):
    all_embeddings = []

    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        inputs = preprocess(batch)
        input_ids = inputs['input_ids'].astype('int32')

        for j in range(len(batch)):
            interpreter.set_tensor(input_details[0]['index'], input_ids[j:j+1])
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])
            all_embeddings.append(output[0])

    return np.array(all_embeddings)

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

corpus_embeddings = embed_batch(texts)

# Simulates how searching could be handled in UNI repo
while True:
    # Potential queries: What freed African Americans, How do animals change over time, Which singer made Thriller
    test = input("Please enter your search query (enter 'q' to quit): ")
    if test.lower() == 'q':
        break

    # Tracks time elapsed, cpu + ram usage
    start_time = time.time()
    ram_before = process.memory_info().rss / 1024 ** 2

    # Embeds user input
    test_embedding = embed_batch([test])

    # Compares similarity between user input and samples
    scores = cosine_similarity(test_embedding, corpus_embeddings)
    results = list(zip(titles, texts, scores[0].tolist()))
    results.sort()

    # Tracks time elapsed, cpu + ram usage
    end_time = time.time() - start_time
    cpu_usage = process.cpu_percent(interval=0.1) / cpu_count
    ram_after = process.memory_info().rss / 1024 ** 2

    count = 0
    # Displays results
    print("\nSimilarity results:")
    for title, text, score in results:
        if score >= 0.7:
            print(f"{score:.4f} - {title}")
            count += 1

    if count < 3:
        for title, text, score in results:
            if score >= 0.65:
                print(f"{score:.4f} - {title}")
                count += 1

    if count < 3:
        for title, text, score in results:
            if score >= 0.6:
                print(f"{score:.4f} - {title}")
                count += 1

    # Displays performance
    print("\nPerformance:")
    print(f"Elapsed time: {end_time:.4f} sec")
    print(f"CPU usage: {cpu_usage:.2f}%")
    print(f"Memory used: {ram_after - ram_before:.2f} MB")