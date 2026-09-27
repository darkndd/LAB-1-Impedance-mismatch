import statistics as stats
import time

import numpy as np
import chromadb


QUERY_IDS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

def main():
    # Conexión a ChromaDB
    client = chromadb.PersistentClient(path="/tmp/chroma_db")
    collection = client.get_collection(name="lab_sentences")
    
    # 1. Recuperamos todos los datos a memoria (Igual que el SELECT de PostgreSQL en P2.py)
    db_data = collection.get(include=["embeddings", "documents"])
    
    if db_data["embeddings"] is None or len(db_data["embeddings"]) == 0:
        raise RuntimeError("No se han encontrado embeddings en ChromaDB")
    
    ids_str = db_data["ids"]
    texts_unsorted = db_data["documents"]
    vectors_unsorted = np.array(db_data["embeddings"], dtype=np.float32)
    
    if len(ids_str) != 10_000:
        raise RuntimeError(f"Se esperaban 10.000 frases con embedding, pero hay {len(ids_str)}")
        
    # Convertir IDs a enteros
    ids_int = np.array([int(id_) for id_ in ids_str])
    
    # Ordenar por ID para mantener el mismo orden que 'ORDER BY id' de PostgreSQL
    sort_idx = np.argsort(ids_int)
    ids = ids_int[sort_idx]
    texts = [texts_unsorted[i] for i in sort_idx]
    vectors = vectors_unsorted[sort_idx]
    
    positions = {int(id_): i for i, id_ in enumerate(ids)}
    
    if any(id_ not in positions for id_ in QUERY_IDS):
        raise RuntimeError("Falta alguna frase de consulta en la tabla")
        
    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms == 0):
        raise RuntimeError("Hay embeddings de norma cero")

    # 2. Computamos las distancias en Numpy para las 2 métricas exigidas
    for metric in ("cosine", "euclidean"):
        times = []
        print(f"\nMétrica (Numpy): {metric}")

        for query_id in QUERY_IDS:
            i = positions[query_id]
            start = time.perf_counter()

            if metric == "cosine":
                distances = 1 - (vectors @ vectors[i]) / (
                    norms * norms[i]
                )
            else:
                distances = np.linalg.norm(vectors - vectors[i], axis=1)

            # Ignoramos la distancia de la frase consigo misma
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

    # 3. BONUS: Query Nativo de ChromaDB (usando su índice vectorial optimizado)
    print("\n" + "="*50)
    print("Métrica: Chroma NATIVE Query (l2 por defecto)")
    
    native_times = []
    
    for query_id in QUERY_IDS:
        i = positions[query_id]
        query_vector = vectors[i].tolist()
        
        start = time.perf_counter()
        
        # Pedimos 3 resultados porque el primero suele ser la propia frase (distancia 0)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        native_times.append(time.perf_counter() - start)
        
        res_ids = results["ids"][0]
        res_distances = results["distances"][0]
        res_docs = results["documents"][0]
        
        print(f"\nConsulta {query_id}: {texts[i]}")
        count = 0
        for r_id, r_dist, r_doc in zip(res_ids, res_distances, res_docs):
            if int(r_id) == query_id:
                continue
            if count >= 2:
                break
            print(f"  Vecino {r_id} (distancia {r_dist:.6f}): {r_doc}")
            count += 1
            
    print(
        f"\nTiempos Chroma NATIVE (s): "
        f"mín={min(native_times):.6f}, "
        f"máx={max(native_times):.6f}, "
        f"media={stats.mean(native_times):.6f}, "
        f"desv={stats.stdev(native_times):.6f}"
    )

if __name__ == "__main__":
    main()
