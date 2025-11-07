from sentence_transformers import SentenceTransformer, util
import psutil
import os
import time

# Loads fine-tuned model, change depending on which model
model = SentenceTransformer("minilm-L6-v2_wp_100_ft")

# Gets system process info to track cpu + ram
process = psutil.Process(os.getpid())
cpu_count = psutil.cpu_count(logical=True)

# Testing sample, change in the future
corpus = [
    "The 13th Amendment abolished slavery, this was after the Civil War.",
    "Evolution is the change in heritable characteristics, initially theorized by Charles Darwin.",
    "Youtube is a website that allows people from around the world to watch and like videos, and subscribe to channels."
]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

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
    test_embedding = model.encode(test, convert_to_tensor=True)

    # Compares similarity between user input and samples
    scores = util.cos_sim(test_embedding, corpus_embeddings)
    results = list(zip(corpus, scores[0].tolist()))

    # Tracks time elapsed, cpu + ram usage
    end_time = time.time() - start_time
    cpu_usage = process.cpu_percent(interval=0.1) / cpu_count
    ram_after = process.memory_info().rss / 1024 ** 2

    # Displays results
    print("\nSimilarity results:")
    for doc, score in results:
        print(f"{score:.4f} - {doc}")

    # Displays performance
    print("\nPerformance:")
    print(f"Elapsed time: {end_time:.4f} sec")
    print(f"CPU usage: {cpu_usage:.2f}%")
    print(f"Memory used: {ram_after - ram_before:.2f} MB")