from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset
from torch.utils.data import DataLoader
import random

# Load dataset
dataset = load_dataset("json", data_files="wikipedia_dataset.jsonl", split="train")
texts = [d["text"] for d in dataset]

# Create training pairs
# [TODO]: Separate text into paragraphs to match with title
tests = [InputExample(texts=[text, text]) for text in texts]
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