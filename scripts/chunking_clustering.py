"""
Chunking, Grouping, Clustering for orchestrator
Groups similar queries for batch processing and memory reuse
"""
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
CLUSTER_PATH = BASE / "runtime" / "logs" / "query_clusters.json"

def chunk_text(text, size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

def micro_chunk_text(text, size=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

def similarity(a, b):
    # Jaccard similarity on words
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def cluster_queries(queries, threshold=0.4):
    clusters = []
    for q in queries:
        placed = False
        for c in clusters:
            if similarity(q, c[0]) > threshold:
                c.append(q)
                placed = True
                break
        if not placed:
            clusters.append([q])
    return clusters

def process_query(query):
    chunks = chunk_text(query)
    micro_chunks = micro_chunk_text(query)
    # Load recent queries for grouping
    from bitdrop_persistence import load_json_bitdrop, save_json
    data = load_json_bitdrop(CLUSTER_PATH)
    recent = []
    if data:
        recent = data.get("recent", [])
    recent.append(query)
    recent = recent[-200:]
    # Main clusters
    clusters = cluster_queries(recent, threshold=0.4)
    # Micro clusters with stricter similarity
    micro_clusters = cluster_queries(recent, threshold=0.7)
    # Micro groups: clusters split further by higher threshold
    micro_groups = []
    for cl in clusters:
        sub = cluster_queries(cl, threshold=0.6)
        micro_groups.extend(sub)
    # Find cluster id for this query
    cluster_id = None
    micro_cluster_id = None
    for i, cl in enumerate(clusters):
        if query in cl:
            cluster_id = i
            break
    for i, cl in enumerate(micro_clusters):
        if query in cl:
            micro_cluster_id = i
            break
    # Save clusters via BitDrop
    save_json(CLUSTER_PATH, {"recent": recent, "clusters": clusters, "micro_clusters": micro_clusters, "micro_groups": micro_groups})
    return {
        "chunks": chunks,
        "micro_chunks": micro_chunks,
        "num_clusters": len(clusters),
        "num_micro_clusters": len(micro_clusters),
        "cluster_size": [len(c) for c in clusters],
        "micro_cluster_size": [len(c) for c in micro_clusters],
        "cluster_id": cluster_id,
        "micro_cluster_id": micro_cluster_id,
        "num_micro_groups": len(micro_groups)
    }

if __name__ == "__main__":
    print(process_query("What is the capital of France and Germany?"))
