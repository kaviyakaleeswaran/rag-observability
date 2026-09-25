from pathlib import Path
from pypdf import PdfReader


DOCS_FOLDER = Path("docs")
OUTPUT_FOLDER = Path("data/extracted")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text


OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


for pdf_file in DOCS_FOLDER.glob("*.pdf"):

    text = extract_text_from_pdf(pdf_file)

    output_file = OUTPUT_FOLDER / f"{pdf_file.stem}.txt"

    output_file.write_text(text, encoding="utf-8")

    print("=" * 60)
    print("Processed:", pdf_file.name)
    print("Pages:", len(PdfReader(pdf_file).pages))
    print("Characters:", len(text))
    print("Saved to:", output_file)
    print("=" * 60)