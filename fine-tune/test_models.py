from sentence_transformers import SentenceTransformer, util
import psutil
import os
import time

# Loads fine-tuned model, change depending on which model
model = SentenceTransformer("minilm-L6-v2_wikipedia100_ft")

# Gets system process info to track cpu + ram
process = psutil.Process(os.getpid())

# Testing sample, change in the future
corpus = [
    "The 13th Amendment abolished slavery.",
    "Evolution is the change in heritable characteristics.",
    "Michael Jackson is a world-renowned pop star."
]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Simulates how searching could be handled in UNI repo
while True:
    # Potential queries: What freed African Americans, How do animals change over time, Which singer made Thriller
    test = input("Please enter your search query (enter 'q' to quit): ")
    if test.lower() == 'q':
        break;

    # Tracks time elapsed, cpu + ram usage
    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    ram_before = process.memory_info().rss / 1024 ** 2

    # Embeds user input
    test_embedding = model.encode(test, convert_to_tensor=True)

    # Compares similarity between user input and samples
    scores = util.cos_sim(test_embedding, corpus_embeddings)
    results = list(zip(corpus, scores[0].tolist()))

    # Tracks time elapsed, cpu + ram usage
    end_time = time.time() - start_time
    cpu_after = process.cpu_percent(interval=None)
    ram_after = process.memory_info().rss / 1024 ** 2

    # Displays results
    print("\nSimilarity results:")
    for doc, score in results:
        print(f"{score:.4f} - {doc}")

    # Displays performance
    print("\nPerformance:")
    print(f"Elapsed time: {end_time:.4f} sec")
    print(f"CPU usage: {cpu_after - cpu_before:.2f}%")
    print(f"Memory used: {ram_after - ram_before:.2f} MB")