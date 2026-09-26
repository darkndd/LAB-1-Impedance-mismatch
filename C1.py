import statistics
import time

import chromadb
from sentence_transformers import SentenceTransformer

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
    # Connexió a Chroma
    client = chromadb.PersistentClient(path="/tmp/chroma_db")
    collection = client.get_collection(name="lab_sentences")

    print("Obtenint dades de ChromaDB...")
    # Obtenim tots els elements de la col·lecció (només volem els documents text)
    db_data = collection.get(include=["documents"])
    ids = db_data["ids"]
    sentences = db_data["documents"]

    if len(ids) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Se esperaban {EXPECTED_ROWS} frases; "
            f"hay {len(ids)} en Chroma"
        )

    print(f"Frases para procesar: {len(sentences)}")
    print(f"Cargando modelo: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    start_generation = time.perf_counter()

    # Generem els embeddings en batch, igual que a P1.py
    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    generation_time = time.perf_counter() - start_generation

    if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIMENSIONS):
        raise RuntimeError(f"Forma inesperada: {embeddings.shape}")

    print(f"Embeddings generados: {embeddings.shape}")
    print(f"Tiempo total de generación: {generation_time:.3f} s")

    storage_times = []
    start_storage = time.perf_counter()

    # Imatgem l'estil de P1.py: fem un UPDATE d'un en un per mesurar el temps per embedding
    print("Iniciant emmagatzematge dels embeddings (update)...")
    for doc_id, embedding in zip(ids, embeddings):
        vector = embedding.tolist()

        start = time.perf_counter()
        
        # A Chroma, utilitzem update() per sobreescriure l'element amb el nou embedding
        collection.update(
            ids=[doc_id],
            embeddings=[vector]
        )

        storage_times.append(time.perf_counter() - start)

    storage_total = time.perf_counter() - start_storage

    print(f"Tiempo total de almacenamiento: {storage_total:.3f} s")
    print_stats("Almacenamiento por embedding", storage_times)


if __name__ == "__main__":
    main()
