"""
Structural Reasoning CLI - PageRank, Betweenness, Communities, Causal Chains
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict, deque
import math

def load_graph(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    concepts = {c['id']: c for c in data.get('concepts', [])}
    label_to_id = {c['label']: c['id'] for c in data.get('concepts', [])}
    relations = data.get('relations', [])
    return concepts, label_to_id, relations

def build_adj(relations, concepts):
    adj = defaultdict(list)
    for r in relations:
        s = r['subject_id']
        o = r['object_id']
        adj[s].append(o)
        adj[o].append(s)
    return adj

def pagerank(concepts, relations, damping=0.85, iters=20):
    adj = build_adj(relations, concepts)
    N = len(concepts)
    scores = {cid: 1.0/N for cid in concepts}
    for _ in range(iters):
        new_scores = {}
        for cid in concepts:
            rank_sum = 0.0
            for nb in adj.get(cid, []):
                # simple undirected
                rank_sum += scores[nb] / max(1, len(adj.get(nb, [])))
            new_scores[cid] = (1-damping)/N + damping * rank_sum
        scores = new_scores
    return scores

def betweenness_like(concepts, relations):
    # Brandes-like approximation for small graphs
    adj = build_adj(relations, concepts)
    betw = {cid: 0.0 for cid in concepts}
    for s in concepts:
        S = []
        P = {v: [] for v in concepts}
        sigma = {v: 0 for v in concepts}
        sigma[s] = 1
        d = {v: -1 for v in concepts}
        d[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in adj.get(v, []):
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = {v: 0.0 for v in concepts}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v]/sigma[w]) * (1 + delta[w])
            if w != s:
                betw[w] += delta[w]
    return betw

def label_propagation(concepts, relations, iters=10):
    adj = build_adj(relations, concepts)
    labels = {cid: cid for cid in concepts}
    for _ in range(iters):
        new_labels = {}
        for cid in concepts:
            neighbors = adj.get(cid, [])
            if not neighbors:
                new_labels[cid] = labels[cid]
                continue
            count = {}
            for nb in neighbors:
                l = labels[nb]
                count[l] = count.get(l, 0) + 1
            # pick most common
            new_labels[cid] = max(count, key=count.get)
        labels = new_labels
    # aggregate communities
    comm = defaultdict(list)
    for cid, lab in labels.items():
        comm[lab].append(cid)
    return comm

def shortest_path(concepts, label_to_id, relations, start_label, end_label):
    if start_label not in label_to_id or end_label not in label_to_id:
        return []
    adj = build_adj(relations, concepts)
    start = label_to_id[start_label]
    end = label_to_id[end_label]
    from collections import deque
    q = deque([(start, [start])])
    seen = {start}
    id_to_label = {v['id']: v['label'] for v in concepts.values()}
    while q:
        node, path = q.popleft()
        if node == end:
            return [id_to_label[n] for n in path]
        for nb in adj.get(node, []):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, path + [nb]))
    return []

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--query", nargs=2, help="start end labels for shortest path")
    args = p.parse_args()
    concepts, label_to_id, relations = load_graph(Path(args.graph))
    # PageRank top 10
    pr = pagerank(concepts, relations)
    id_to_label = {v['id']: v['label'] for v in concepts.values()}
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top PageRank:")
    for cid, score in top_pr:
        print(f"  {id_to_label[cid]}: {score:.4f}")
    # Betweenness
    betw = betweenness_like(concepts, relations)
    top_b = sorted(betw.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop Betweenness-like:")
    for cid, score in top_b:
        print(f"  {id_to_label[cid]}: {score:.2f}")
    # Communities
    comm = label_propagation(concepts, relations)
    print(f"\nCommunities: {len(comm)}")
    for i, (lab, members) in enumerate(list(comm.items())[:5]):
        labels = [id_to_label[c] for c in members[:5]]
        print(f"  Community {i}: {labels} ...")
    # Shortest path
    if args.query:
        path = shortest_path(concepts, label_to_id, relations, args.query[0], args.query[1])
        print(f"\nShortest path {args.query[0]} -> {args.query[1]}:")
        print(" -> ".join(path) if path else "No path")

if __name__ == "__main__":
    main()
