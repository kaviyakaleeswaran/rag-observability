from pypdf import PdfReader

pdf_path = "docs/vit.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print("Characters extracted:", len(text))

print("\n--- FIRST 2000 CHARACTERS ---\n")
print(text[:2000])