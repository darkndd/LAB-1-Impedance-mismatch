import csv
import hashlib
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "bookcorpus_10000.csv"
OUTPUT_PATH = BASE_DIR / "bookcorpus_embeddings.npz"

EXPECTED_ROWS = 10_000
EXPECTED_DIMENSIONS = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH.name} ya existe. No se sobrescribirá."
        )

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not {"id", "sentence"}.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Columnas esperadas: id,sentence. "
                f"Encontradas: {reader.fieldnames}"
            )

        rows = [
            (int(row["id"]), row["sentence"])
            for row in reader
        ]

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Se esperaban {EXPECTED_ROWS} frases; hay {len(rows)}"
        )

    ids = np.asarray([row[0] for row in rows], dtype=np.int32)
    sentences = [row[1] for row in rows]

    if len(set(ids.tolist())) != EXPECTED_ROWS:
        raise ValueError("El CSV contiene IDs repetidos")

    if any(not sentence.strip() for sentence in sentences):
        raise ValueError("El CSV contiene frases vacías")

    csv_sha256 = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

    print("Cargando modelo...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    start = time.perf_counter()

    embeddings = model.encode(
        sentences,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    generation_time = time.perf_counter() - start

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIMENSIONS):
        raise RuntimeError(
            f"Dimensiones inesperadas: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Se encontraron valores NaN o infinitos")

    np.savez_compressed(
        OUTPUT_PATH,
        ids=ids,
        embeddings=embeddings,
        csv_sha256=np.asarray(csv_sha256),
    )

    print(f"Vectores generados: {embeddings.shape}")
    print(f"Tiempo de generación: {generation_time:.3f} s")
    print(f"Archivo creado: {OUTPUT_PATH.name}")
    print(f"SHA-256 del CSV: {csv_sha256}")


if __name__ == "__main__":
    main()