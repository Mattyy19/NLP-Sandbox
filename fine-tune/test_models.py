from sentence_transformers import SentenceTransformer, util

# Loads fine-tuned model, change depending on which model
model = SentenceTransformer("minilm-L6-v2_wikipedia100_ft")

# Testing sample, change in the future
corpus = [
    "The 13th Amendment abolished slavery.",
    "Evolution is the change in heritable characteristics.",
    "Michael Jackson is a world-renowned pop star."
]

# Simulates how searching could be handled in UNI repo
while True:
    # Potential queries: What freed African Americans, How do animals change over time, Which singer made Thriller
    test = input("Please enter your search query: ")

    # Embeds user input and samples
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    test_embedding = model.encode(test, convert_to_tensor=True)

    # Compares similarity between user input and samples
    scores = util.cos_sim(test_embedding, corpus_embeddings)
    results = list(zip(corpus, scores[0].tolist()))

    # Displays results
    for doc, score in results:
        print(f"{score:.4f} - {doc}")