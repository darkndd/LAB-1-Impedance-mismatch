import csv
import statistics
import time
from pathlib import Path

import psycopg2

from config import load_config


CSV_PATH = Path(__file__).resolve().parent / "bookcorpus_10000.csv"
EXPECTED_ROWS = 10_000


def load_csv():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not {"id", "sentence"}.issubset(reader.fieldnames or []):
            raise ValueError(
                f"El CSV debe tener columnas id,sentence; "
                f"encontradas: {reader.fieldnames}"
            )

        rows = [
            (int(row["id"]), row["sentence"].strip())
            for row in reader
        ]

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Se esperaban {EXPECTED_ROWS} frases, pero hay {len(rows)}"
        )

    if any(not sentence for _, sentence in rows):
        raise ValueError("Hay frases vacías en el CSV")

    if len({sentence_id for sentence_id, _ in rows}) != len(rows):
        raise ValueError("Hay IDs repetidos en el CSV")

    return rows


def main():
    rows = load_csv()

    connection = psycopg2.connect(**load_config())

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lab_sentences (
                    id INTEGER PRIMARY KEY,
                    sentence TEXT NOT NULL,
                    embedding REAL[]
                )
            """)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM lab_sentences")
            existing_rows = cursor.fetchone()[0]

        if existing_rows != 0:
            raise RuntimeError(
                f"lab_sentences ya tiene {existing_rows} filas. "
                "P0 se detiene para evitar duplicados o mezclar experimentos."
            )

        times = []
        start_total = time.perf_counter()

        with connection.cursor() as cursor:
            for sentence_id, sentence in rows:
                start = time.perf_counter()

                cursor.execute(
                    """
                    INSERT INTO lab_sentences (id, sentence)
                    VALUES (%s, %s)
                    """,
                    (sentence_id, sentence),
                )
                connection.commit()

                times.append(time.perf_counter() - start)

            cursor.execute("SELECT COUNT(*) FROM lab_sentences")
            count = cursor.fetchone()[0]

        total = time.perf_counter() - start_total

        print(f"Frases guardadas en PostgreSQL: {count}")
        print(f"Tiempo total del bucle: {total:.3f} s")
        print(f"Tiempo mínimo por frase: {min(times):.6f} s")
        print(f"Tiempo máximo por frase: {max(times):.6f} s")
        print(f"Tiempo medio por frase: {statistics.mean(times):.6f} s")
        print(
            "Desviación estándar por frase: "
            f"{statistics.stdev(times):.6f} s"
        )

        if count != EXPECTED_ROWS:
            raise RuntimeError(
                f"Se esperaban {EXPECTED_ROWS} filas, pero hay {count}"
            )

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()