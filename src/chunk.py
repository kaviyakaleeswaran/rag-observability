from pathlib import Path
import json


INPUT_FOLDER = Path("data/extracted")
OUTPUT_FOLDER = Path("data")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def create_chunks(text):
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        start = end - CHUNK_OVERLAP

    return chunks


all_chunks = []

for text_file in INPUT_FOLDER.glob("*.txt"):

    text = text_file.read_text(encoding="utf-8")

    chunks = create_chunks(text)

    for index, chunk in enumerate(chunks):

        all_chunks.append({
            "chunk_id": f"{text_file.stem}_{index}",
            "source": text_file.stem,
            "text": chunk
        })

    print(f"{text_file.name}: {len(chunks)} chunks")


output_file = OUTPUT_FOLDER / "chunks.json"

output_file.write_text(
    json.dumps(all_chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print()
print("Total chunks:", len(all_chunks))
print("Saved to:", output_file)