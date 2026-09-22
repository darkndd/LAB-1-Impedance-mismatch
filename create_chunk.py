from datasets import load_dataset
import csv
import re

TARGET_SENTENCES = 10_000
OUTPUT_FILE = "bookcorpus_10000.csv"

print("Cargando dataset SamuelYang/bookcorpus...")

dataset = load_dataset(
    "SamuelYang/bookcorpus",
    split="train"
)

print(f"Ejemplos cargados: {len(dataset)}")
print(f"Columnas disponibles: {dataset.column_names}")

sentences = []

for row in dataset:
    text = row["text"]

    # Normaliza espacios y saltos de línea
    text = re.sub(r"\s+", " ", text).strip()

    # Divide por signos de final de frase
    parts = re.split(r"(?<=[.!?])\s+", text)

    for sentence in parts:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentences.append(sentence)

        if len(sentences) >= TARGET_SENTENCES:
            break

    if len(sentences) >= TARGET_SENTENCES:
        break

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "sentence"])

    for sentence_id, sentence in enumerate(sentences):
        writer.writerow([sentence_id, sentence])

print(f"Frases guardadas: {len(sentences)}")
print(f"Archivo generado: {OUTPUT_FILE}")