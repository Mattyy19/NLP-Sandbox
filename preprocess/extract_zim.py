import json
import re
from bs4 import BeautifulSoup
from pyzim import Zim
import pyzim.compression

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\fine-tune\pm-minilm-L12-v2_wp_all_lang_v2.11_ft")
tokenizer = embed_model.tokenizer

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
OUTPUT_FILE = r"C:\Users\Matthew\IdeaProjects\NLP-Sandbox\fine-tune\wp_paragraph.jsonl"

def token_chunk(text, chunk_size=256, overlap=64):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks

# Removes html tags, reference numbers and large whitespaces
def normalize(text):
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sections(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    sections = []

    # Try to match Java logic (.mw-parser-output)
    root = soup.select_one(".mw-parser-output")
    if root is None:
        root = soup.body

    # Remove unwanted elements
    for tag in root.select("script, style, sup, table, .navbox, .infobox, .thumb, .metadata"):
        tag.decompose()

    current_title = "Introduction"
    current_text = []

    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if el.name.startswith("h"):  # heading
            # Save previous section
            if current_text:
                text = normalize(" ".join(current_text))
                if len(text) >= 100:
                    sections.append({
                        "title": current_title,
                        "text": text
                    })

            current_title = el.get_text(strip=True)
            current_text = []
        else:
            text = el.get_text(strip=True)
            if text:
                current_text.append(text)

    # Save final section
    if current_text:
        text = normalize(" ".join(current_text))
        if len(text) >= 100:
            sections.append({
                "title": current_title,
                "text": text
            })

    return sections

def extract_paragraphs(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    sections = []

    # Try to match Java logic (.mw-parser-output)
    root = soup.select_one(".mw-parser-output")
    if root is None:
        root = soup.body

    # Remove unwanted elements
    for tag in root.select("script, style, sup, table, .navbox, .infobox, .thumb, .metadata"):
        tag.decompose()

    current_title = "Introduction"
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if el.name.startswith("h"):  # heading
            current_title = el.get_text(strip=True)
        elif el.name in ["p", "li"]:
            text = el.get_text(strip=True)
            text = normalize(text)
            sections.append({
                "title": current_title,
                "text": text
            })

    return sections

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
                    section_text = extract_paragraphs(raw_text)
                except Exception as e:
                    print(f"Error whhile trying to clean text: {e}")
                    continue

                # Writes to jsonl file
                for section in section_text:
                    section_title = f"{entry.title} - {section['title']}"
                    chunks = [section["text"]]

                    for i, chunk in enumerate(chunks):
                        out_file.write(json.dumps({
                            "title": section_title,
                            "section": section["title"],
                            "chunk_id": i,
                            "text": chunk
                        }, ensure_ascii=False) + "\n")

                    print(f"Title: {section_title}, Length: {len(section['text'])}")
                    dataset_count += 1

            except Exception as e:
                print(f"Entry: {entry.title} has an error")
                continue

print(f"Total entries written: {dataset_count}")
