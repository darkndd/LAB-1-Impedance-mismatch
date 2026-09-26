import statistics as stats
import time

import numpy as np
import psycopg2

from config import load_config


QUERY_IDS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

with psycopg2.connect(**load_config()) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, sentence, embedding "
            "FROM lab_sentences ORDER BY id"
        )
        rows = cur.fetchall()

if len(rows) != 10_000 or any(row[2] is None for row in rows):
    raise RuntimeError("Se esperaban 10.000 frases con embedding")

ids = np.array([row[0] for row in rows])
texts = [row[1] for row in rows]
vectors = np.array([row[2] for row in rows], dtype=np.float32)
positions = {int(id_): i for i, id_ in enumerate(ids)}

if any(id_ not in positions for id_ in QUERY_IDS):
    raise RuntimeError("Falta alguna frase de consulta en la tabla")

norms = np.linalg.norm(vectors, axis=1)

if np.any(norms == 0):
    raise RuntimeError("Hay embeddings de norma cero")

for metric in ("cosine", "euclidean"):
    times = []
    print(f"\nMétrica: {metric}")

    for query_id in QUERY_IDS:
        i = positions[query_id]
        start = time.perf_counter()

        if metric == "cosine":
            distances = 1 - (vectors @ vectors[i]) / (
                norms * norms[i]
            )
        else:
            distances = np.linalg.norm(vectors - vectors[i], axis=1)

        distances[i] = np.inf
        nearest = np.argsort(distances)[:2]
        times.append(time.perf_counter() - start)

        print(f"\nConsulta {query_id}: {texts[i]}")
        for j in nearest:
            print(
                f"  Vecino {ids[j]} "
                f"(distancia {distances[j]:.6f}): {texts[j]}"
            )

    print(
        f"\nTiempos {metric} (s): "
        f"mín={min(times):.6f}, "
        f"máx={max(times):.6f}, "
        f"media={stats.mean(times):.6f}, "
        f"desv={stats.stdev(times):.6f}"
    )