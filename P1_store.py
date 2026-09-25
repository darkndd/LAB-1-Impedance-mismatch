import csv
import hashlib
import statistics
import time
from pathlib import Path

import numpy as np
import psycopg2

from config import load_config


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "bookcorpus_10000.csv"
EMBEDDINGS_PATH = BASE_DIR / "bookcorpus_embeddings.npz"

EXPECTED_ROWS = 10_000
EXPECTED_DIMENSIONS = 384


def print_stats(label, times):
    print(f"\n{label} ({len(times)} muestras, segundos)")
    print(f"Mínimo: {min(times):.6f}")
    print(f"Máximo: {max(times):.6f}")
    print(f"Media: {statistics.mean(times):.6f}")
    print(f"Desviación estándar: {statistics.stdev(times):.6f}")


def load_expected_ids():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return [int(row["id"]) for row in reader]


def main():
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {EMBEDDINGS_PATH.name}. "
            "Ejecuta primero P1_generate.py en WSL."
        )

    current_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    expected_ids = load_expected_ids()

    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as data:
        ids = data["ids"]
        embeddings = data["embeddings"]
        saved_hash = str(data["csv_sha256"].item())

    if saved_hash != current_hash:
        raise RuntimeError(
            "El CSV ha cambiado desde que se generaron los embeddings"
        )

    if len(expected_ids) != EXPECTED_ROWS:
        raise RuntimeError(
            f"El CSV tiene {len(expected_ids)} frases, no {EXPECTED_ROWS}"
        )

    if ids.tolist() != expected_ids:
        raise RuntimeError(
            "El orden o los IDs de los embeddings no coinciden con el CSV"
        )

    if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIMENSIONS):
        raise RuntimeError(
            f"Forma inesperada de embeddings: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Hay valores NaN o infinitos")

    connection = psycopg2.connect(**load_config())

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM lab_sentences"
            )
            row_count = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM lab_sentences
                WHERE embedding IS NOT NULL
                """
            )
            existing_embeddings = cursor.fetchone()[0]

        if row_count != EXPECTED_ROWS:
            raise RuntimeError(
                f"PostgreSQL tiene {row_count} frases, "
                f"no {EXPECTED_ROWS}"
            )

        if existing_embeddings:
            raise RuntimeError(
                f"Ya hay {existing_embeddings} embeddings. "
                "No se sobrescribirán."
            )

        print(f"Frases en PostgreSQL: {row_count}")
        print(f"Vectores para guardar: {len(ids)}")

        storage_times = []
        start_total = time.perf_counter()

        with connection.cursor() as cursor:
            for sentence_id, embedding in zip(ids, embeddings):
                start = time.perf_counter()

                cursor.execute(
                    """
                    UPDATE lab_sentences
                    SET embedding = %s
                    WHERE id = %s AND embedding IS NULL
                    """,
                    (embedding.tolist(), int(sentence_id)),
                )

                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"No se ha actualizado exactamente "
                        f"una fila para id={sentence_id}"
                    )

                connection.commit()
                storage_times.append(time.perf_counter() - start)

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM lab_sentences
                WHERE embedding IS NOT NULL
                """
            )
            stored_count = cursor.fetchone()[0]

        if stored_count != EXPECTED_ROWS:
            raise RuntimeError(
                f"Solo hay {stored_count} embeddings almacenados"
            )

        print(f"Embeddings almacenados: {stored_count}")
        print(f"Tiempo total del bucle: "
              f"{time.perf_counter() - start_total:.3f} s")
        print_stats("Almacenamiento por embedding", storage_times)

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()