from sentence_transformers import SentenceTransformer, util
import psutil
import os
import time
import json

# Loads fine-tuned model, change depending on which model
model = SentenceTransformer("pm-minilm-L12-v2_wp_all_lang_ft")

# Gets system process info to track cpu + ram
process = psutil.Process(os.getpid())
cpu_count = psutil.cpu_count(logical=True)

# Testing sample, change in the future
titles = []
sections = []
chunk_ids = []
texts = []
file_path = r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\fine-tune\wikipedia_dataset.jsonl"

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
    section = data.get("section", "")
    chunk_id = data.get("chunk_id", 0)
    text = data.get("text", "")

    if text:
        titles.append(title)
        sections.append(section)
        chunk_ids.append(chunk_id)
        texts.append(text)

corpus_embeddings = model.encode(
    texts,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True
)

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
    test_embedding = model.encode([test], convert_to_tensor=True)

    # Compares similarity between user input and samples
    scores = util.cos_sim(test_embedding, corpus_embeddings)
    results = list(zip(titles, texts, scores[0].tolist()))

    # Tracks time elapsed, cpu + ram usage
    end_time = time.time() - start_time
    cpu_usage = process.cpu_percent(interval=0.1) / cpu_count
    ram_after = process.memory_info().rss / 1024 ** 2

    # Displays results
    print("\nSimilarity results:")
    for title, text, score in results:
        if score >= 0.5:
            print(f"{score:.4f} - {title}")

    # Displays performance
    print("\nPerformance:")
    print(f"Elapsed time: {end_time:.4f} sec")
    print(f"CPU usage: {cpu_usage:.2f}%")
    print(f"Memory used: {ram_after - ram_before:.2f} MB")