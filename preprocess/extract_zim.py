import json
import re
from bs4 import BeautifulSoup
from pyzim import Zim
import pyzim.compression

# Enable Zstandard support
try:
    import zstandard
    pyzim.compression.CompressionRegistry.register(
        pyzim.compression.CompressionType.ZSTD,
        pyzim.compression.ZstandardCompressionInterface)
except ImportError:
    print("Warning: zstandard not installed. ZIMs with type 5 compression will fail.")


# Config (change paths to match your structure)
ZIM_FILE = r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\wikipedia_en_100_nopic_2025-09.zim"
OUTPUT_FILE = r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\wikipedia_dataset.jsonl"

# Removes html tags, reference numbers and large whitespaces
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text()
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Extracts content from zim files and writes to a jsonl file
dataset_count = 0
with open(ZIM_FILE, "rb") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
    with Zim(f) as zim:
        for entry in zim.iter_entries():
            # C is main namespace for articles, will skip if it isn't
            if entry.namespace not in ("C",):
                continue
            # Skips if it isn't an article
            if not entry.is_article:
                #print(entry.title)
                continue
            # Skips if it is a redirect
            if entry.is_redirect:
                continue
            # Skips main page of zim
            if entry.title == "Main Page":
                continue

            try:
                # Skips non-text content
                mimetype = getattr(entry, "mimetype", "")
                if not mimetype.startswith("text/") and not mimetype.startswith("application/xhtml"):
                    continue

                # Read and decode content
                raw_bytes = entry.read()
                if not raw_bytes:
                    continue
                raw_text = raw_bytes.decode("utf-8", errors="ignore")

                # Cleans text
                try:
                    cleaned_text = clean_html(raw_text)
                except Exception:
                    print("Error whhile trying to clean text")
                    continue

                # Skip short text
                if len(cleaned_text) < 100:
                    continue

                # Writes to jsonl file
                print(f"Title: {entry.title}, Length: {len(cleaned_text)}")
                out_file.write(json.dumps({"title": entry.title, "text": cleaned_text}, ensure_ascii=False) + "\n")
                dataset_count += 1

            except Exception as e:
                print(f"Entry: {entry.title} has an error")
                continue

print(f"Total entries written: {dataset_count}")
