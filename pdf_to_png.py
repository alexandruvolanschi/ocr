from pdf2image import convert_from_path

# Calea către PDF
pdf_path = r"D:\mastere.pdf"

# Calea către Poppler
poppler_path = r"D:\poppler\bin\poppler-24.08.0\Library\bin"

# Conversie PDF în imagini
images = convert_from_path(pdf_path, dpi=600, poppler_path=poppler_path)

# Salvare fiecare pagină ca PNG
for i, img in enumerate(images):
    output_path = f"D:\\master_{i+1}.png"
    img.save(output_path, "PNG")
    print(f"Salvat: {output_path}")
