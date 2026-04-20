from sentence_transformers import SentenceTransformer, util
import psutil
import os
import time

# For translation
import langid
from transformers import MarianMTModel, MarianTokenizer

langid.set_languages(["en", "es", "fr"])


# Detects what the language is
def detect_lang(src_text):
    return langid.classify(src_text)[0]


# Selects model based on the source language and target language
def select_model(src_lang, tgt_lang):
    if src_lang == "es" and tgt_lang == "en":
        return "Helsinki-NLP/opus-mt-es-en"
    if src_lang == "fr" and tgt_lang == "en":
        return "Helsinki-NLP/opus-mt-fr-en"


# Translates based on the source language and target language
def translate(src_lang, tgt_lang, src_text):
    # Selects model to use for translation
    model_name = select_model(src_lang, tgt_lang)
    tok = MarianTokenizer.from_pretrained(model_name)
    translate_model = MarianMTModel.from_pretrained(model_name)

    # Translates
    inputs = tok(src_text, return_tensors="pt", padding=True)
    translated_tokens = translate_model.generate(**inputs)
    translated_text = tok.decode(translated_tokens[0], skip_special_tokens=True)

    del translate_model
    del tok

    return translated_text


# Loads fine-tuned model, change depending on which model
model = SentenceTransformer("minilm-L6-v2_wp_100_chem_math_phys_ft")

# Gets system process info to track cpu + ram
process = psutil.Process(os.getpid())
cpu_count = psutil.cpu_count(logical=True)

# Testing sample, change in the future
corpus = [
    "La 13ª Enmienda abolió la esclavitud, esto ocurrió después de la Guerra Civil.",
    "L'évolution est le changement des caractéristiques héréditaires, initialement théorisée par Charles Darwin.",
    "Youtube is a website that allows people from around the world to watch and like videos, and subscribe to channels."
]

# English version that will be embedded
english_corpus = []

# Converts articles to English
for text in corpus:
    lang = detect_lang(text)

    # Adds article if it is already in English
    if lang == "en":
        english_corpus.append(text)
        continue

    # Translates and adds to English version of corpus
    english_corpus.append(translate(lang, "en", text))

# for text in english_corpus:
#     print(text)

corpus_embeddings = model.encode(english_corpus, convert_to_tensor=True)

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
    user_lang = detect_lang(test)
    if user_lang == "en":
        test_embedding = model.encode(test, convert_to_tensor=True)
    else:
        test_embedding = model.encode(translate(user_lang, "en", test), convert_to_tensor=True)

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
