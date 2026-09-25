import statistics
import time

import psycopg2
from sentence_transformers import SentenceTransformer

from config import load_config


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
EXPECTED_ROWS = 10_000
EXPECTED_DIMENSIONS = 384


def print_stats(name, values):
    print(f"\n{name} ({len(values)} muestras, segundos)")
    print(f"Mínimo: {min(values):.6f}")
    print(f"Máximo: {max(values):.6f}")
    print(f"Media: {statistics.mean(values):.6f}")
    print(f"Desviación estándar: {statistics.stdev(values):.6f}")


def main():
    connection = psycopg2.connect(**load_config())

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, sentence
                FROM lab_sentences
                ORDER BY id
            """)
            rows = cursor.fetchall()

            cursor.execute("""
                SELECT COUNT(*)
                FROM lab_sentences
                WHERE embedding IS NOT NULL
            """)
            already_embedded = cursor.fetchone()[0]

        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Se esperaban {EXPECTED_ROWS} frases; "
                f"hay {len(rows)} en PostgreSQL"
            )

        if already_embedded != 0:
            raise RuntimeError(
                f"Ya hay {already_embedded} embeddings almacenados. "
                "P1 se detiene para evitar sobrescribirlos."
            )

        print(f"Frases para procesar: {len(rows)}")
        print(f"Cargando modelo: {MODEL_NAME}")

        model = SentenceTransformer(MODEL_NAME, device="cpu")

        sentences = [sentence for _, sentence in rows]

        start_generation = time.perf_counter()

        embeddings = model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        generation_time = time.perf_counter() - start_generation

        if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIMENSIONS):
            raise RuntimeError(
                f"Forma inesperada: {embeddings.shape}"
            )

        print(f"Embeddings generados: {embeddings.shape}")
        print(f"Tiempo total de generación: {generation_time:.3f} s")

        storage_times = []
        start_storage = time.perf_counter()

        with connection.cursor() as cursor:
            for (sentence_id, _), embedding in zip(rows, embeddings):
                vector = embedding.tolist()

                start = time.perf_counter()

                cursor.execute(
                    """
                    UPDATE lab_sentences
                    SET embedding = %s
                    WHERE id = %s
                    """,
                    (vector, sentence_id),
                )
                connection.commit()

                storage_times.append(time.perf_counter() - start)

            cursor.execute("""
                SELECT COUNT(*)
                FROM lab_sentences
                WHERE embedding IS NOT NULL
            """)
            stored_count = cursor.fetchone()[0]

        storage_total = time.perf_counter() - start_storage

        if stored_count != EXPECTED_ROWS:
            raise RuntimeError(
                f"Solo hay {stored_count} embeddings almacenados"
            )

        print(f"Embeddings almacenados: {stored_count}")
        print(f"Tiempo total de almacenamiento: {storage_total:.3f} s")
        print_stats("Almacenamiento por embedding", storage_times)

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()