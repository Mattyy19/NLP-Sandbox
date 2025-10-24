from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt_tab")

# Load dataset
dataset = load_dataset("json", data_files="wikipedia_dataset.jsonl", split="train")

# Creates training pairs for fine tuning
tests = []
for data in dataset:
    try:
        title = data.get("title", "")
        text = data.get("text", "")

        # Creates sentences out of the text, then every 3 sentences makes it a pair with the title
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences), 3):
            para = " ".join(sentences[i:i+3]).strip()
            # Skips if paragraph is too short
            if len(para) > 100:
                tests.append(InputExample(texts=[title, para]))

    except Exception as e:
        print("Something went wrong with creating training pairs :(")
        print(e)
        continue
random.shuffle(tests)

# Load model and prepare for fine tuning
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
train_dataloader = DataLoader(tests, shuffle=True, batch_size=8, num_workers=0)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine tune model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

# Save model
model.save("minilm-L6-v2_wikipedia100_ft")
print("Model saved")