import csv
import statistics
import time
from pathlib import Path

# Pas 2: Importem ChromaDB en lloc de psycopg2
import chromadb


CSV_PATH = Path(__file__).resolve().parent / "bookcorpus_10000.csv"
EXPECTED_ROWS = 10_000


def load_csv():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = [
            (str(row["id"]), row["sentence"].strip()) # Chroma requereix IDs en format Text (String)
            for row in reader
        ]
    return rows


def main():
    rows = load_csv()

    # Connexió a ChromaDB de forma local (crearà una carpeta 'chroma_db')
    client = chromadb.PersistentClient(path="/tmp/chroma_db")
    
    # Creem la col·lecció (equivalent a CREATE TABLE de PostgreSQL)
    collection = client.get_or_create_collection(name="lab_sentences")

    if collection.count() != 0:
        raise RuntimeError(
            f"La col·lecció 'lab_sentences' ja té {collection.count()} elements. "
            "C0 es detura per evitar duplicats."
        )

    times = []
    start_total = time.perf_counter()

    # Inserim el text a ChromaDB frase per frase, mesurant el temps
    for sentence_id, sentence in rows:
        start = time.perf_counter()

        collection.add(
            ids=[sentence_id],
            documents=[sentence]
        )

        times.append(time.perf_counter() - start)

    count = collection.count()
    total = time.perf_counter() - start_total

    print(f"Frases guardades a ChromaDB: {count}")
    print(f"Tiempo total del bucle: {total:.3f} s")
    print(f"Tiempo mínimo por frase: {min(times):.6f} s")
    print(f"Tiempo máximo por frase: {max(times):.6f} s")
    print(f"Tiempo medio por frase: {statistics.mean(times):.6f} s")
    print(f"Desviación estándar por frase: {statistics.stdev(times):.6f} s")


if __name__ == "__main__":
    main()
