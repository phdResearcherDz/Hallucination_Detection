import os
import json
import time
import traceback
import regex as re
import numpy as np
import networkx as nx

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


############################################################
# CONFIG
############################################################

GRAPH_DIR    = "graph_outputs_not_labled"
FEATURE_DIR  = "features_not_labeled"
OUTFILE      = os.path.join(FEATURE_DIR, "features.jsonl")

os.makedirs(FEATURE_DIR, exist_ok=True)

DATASET_NAME   = "UTAustin-AIHealth/MedHallu"
DATASET_CONFIG = "pqa_artificial"
DATASET_SPLIT  = "train"

# Embeddings = same model as Part 1
EMBEDDER = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
EMB_DIM  = EMBEDDER.get_sentence_embedding_dimension()


############################################################
# EMBEDDING UTILITIES
############################################################

def embed(text):
    return EMBEDDER.encode(text, convert_to_numpy=True)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def best_sim(vec, vecs):
    if not vecs:
        return 0.0
    return max(cosine(vec, u) for u in vecs)


############################################################
# LOAD GRAPHS FROM DISK
############################################################

def load_graph(json_graph):
    """Load graph stored in Part 1 format."""
    G = nx.DiGraph()
    for n in json_graph["nodes"]:
        G.add_node(n["id"], text=n["text"], vec=np.array(n["vec"]))
    for e in json_graph["edges"]:
        G.add_edge(
            e["head"], e["tail"],
            relation=e["relation"],
            text=e["text"],
            vec=np.array(e["vec"])
        )
    return G


############################################################
# GRAPH FEATURE HELPERS
############################################################

def node_vectors(G):
    return [G.nodes[n]["vec"] for n in G.nodes()]

def avg_graph_vec(G):
    """Graph embedding = mean of node embeddings."""
    if G.number_of_nodes() == 0:
        return np.zeros(EMB_DIM)
    return np.mean([G.nodes[n]["vec"] for n in G.nodes()], axis=0)

def relation_vectors(G):
    return [G.edges[u, v]["vec"] for u, v in G.edges()]

def compute_edge_coherence(G):
    """
    For each edge (h, r, t), compute average cosine of:
    cos(h_vec, r_vec), cos(r_vec, t_vec), cos(h_vec, t_vec)
    """
    if G.number_of_edges() == 0:
        return 0.0

    scores = []
    for u, v in G.edges():
        h_vec = G.nodes[u]["vec"]
        t_vec = G.nodes[v]["vec"]
        r_vec = G.edges[u, v]["vec"]

        c1 = cosine(h_vec, r_vec)
        c2 = cosine(r_vec, t_vec)
        c3 = cosine(h_vec, t_vec)
        scores.append((c1 + c2 + c3) / 3.0)

    return float(np.mean(scores))


############################################################
# ADVANCED FEATURE ENGINEERING
############################################################

def compute_alignment_features(Ga, Gref):
    """
    Alignment between answer graph (Ga) & reference graph (Gref) (Q or K)
    Returns dict of node-level + edge-level alignment metrics.
    """

    feats = {}

    # --- Node alignment ---
    A_nodes = node_vectors(Ga)
    R_nodes = node_vectors(Gref)

    if A_nodes and R_nodes:
        node_sims = [best_sim(v, R_nodes) for v in A_nodes]
        feats["node_align_mean"] = float(np.mean(node_sims))
        feats["node_align_min"]  = float(np.min(node_sims))
        feats["node_align_frac_low"] = float(np.mean([1 if s < 0.45 else 0 for s in node_sims]))
    else:
        feats["node_align_mean"] = 0
        feats["node_align_min"]  = 0
        feats["node_align_frac_low"] = 1.0

    # --- Edge alignment ---
    A_rels = relation_vectors(Ga)
    R_rels = relation_vectors(Gref)

    if A_rels and R_rels:
        rel_sims = [best_sim(vec, R_rels) for vec in A_rels]
        feats["rel_align_mean"] = float(np.mean(rel_sims))
        feats["rel_align_min"]  = float(np.min(rel_sims))
        feats["rel_align_frac_low"] = float(np.mean([1 if s < 0.45 else 0 for s in rel_sims]))
    else:
        feats["rel_align_mean"] = 0
        feats["rel_align_min"]  = 0
        feats["rel_align_frac_low"] = 1.0

    return feats


def compute_graph_features(Ga, Gq, Gk, question, knowledge, answer):
    """Compute all features for a single answer graph."""

    feats = {}

    # ----------------------------------------------------
    # TEXT-LEVEL FEATURES
    # ----------------------------------------------------
    vq = embed(question)
    vk = embed(knowledge)
    va = embed(answer)

    feats["sim_a_q"] = cosine(va, vq)
    feats["sim_a_k"] = cosine(va, vk)

    # Lexical Jaccard
    set_q = set(question.lower().split())
    set_k = set(knowledge.lower().split())
    set_a = set(answer.lower().split())

    feats["jaccard_a_q"] = len(set_a & set_q) / (len(set_a | set_q) + 1e-9)
    feats["jaccard_a_k"] = len(set_a & set_k) / (len(set_a | set_k) + 1e-9)

    # ----------------------------------------------------
    # GRAPH-LEVEL EMBEDDINGS
    # ----------------------------------------------------
    gq = avg_graph_vec(Gq)
    gk = avg_graph_vec(Gk)
    ga = avg_graph_vec(Ga)

    feats["graph_sim_a_q"] = cosine(ga, gq)
    feats["graph_sim_a_k"] = cosine(ga, gk)

    # ----------------------------------------------------
    # STRUCTURE FEATURES
    # ----------------------------------------------------
    feats["nodes_a"] = Ga.number_of_nodes()
    feats["edges_a"] = Ga.number_of_edges()
    feats["density_a"] = nx.density(Ga) if Ga.number_of_nodes() > 1 else 0

    feats["coherence_a"] = compute_edge_coherence(Ga)

    return feats


############################################################
# PROCESS ONE SAMPLE
############################################################

def process_sample(i, sample):
    graph_path = os.path.join(GRAPH_DIR, f"sample_{i:04d}.json")
    if not os.path.exists(graph_path):
        print(f"⚠ Missing graph: {graph_path}")
        return []

    gjson = json.load(open(graph_path, "r", encoding="utf-8"))

    Gq = load_graph(gjson["graph_question"])
    Gk = load_graph(gjson["graph_knowledge"])
    Ga = load_graph(gjson["graph_answer"])   # already normalized in Part 1!

    question   = sample["Question"]
    knowledge  = " ".join(sample["Knowledge"])
    hallucinated = sample["Hallucinated Answer"]
    gt_answer    = sample["Ground Truth"]
    category     = sample["Category of Hallucination"]

    rows = []

    # ========================================================
    # HALLUCINATED ANSWER (label = 1)
    # ========================================================
    print(f"   Computing features for HALLUCINATED answer...")
    feats_h = {}

    feats_h.update(compute_graph_features(Ga, Gq, Gk,
                                          question, knowledge, hallucinated))

    # alignment w.r.t question & knowledge
    Qa = compute_alignment_features(Ga, Gq)
    Ka = compute_alignment_features(Ga, Gk)

    for k, v in Qa.items(): feats_h[f"Q_{k}"] = v
    for k, v in Ka.items(): feats_h[f"K_{k}"] = v

    feats_h["binary_label"] = 1
    feats_h["category"] = category

    rows.append(feats_h)

    # ========================================================
    # GROUND TRUTH ANSWER (label = 0)
    # We reuse the same graphs: rebuild answer graph for GT
    # ========================================================

    print(f"   Computing features for GROUND TRUTH answer…")
    # Build GT graph here via Part1-like process
    # BUT: If you prefer, you can also generate GT graph in Part1.
    from graph_builder_utils import build_graph  # Create helper or reuse Part1

    Ggt = build_graph(gt_answer)

    feats_gt = {}
    feats_gt.update(compute_graph_features(Ggt, Gq, Gk,
                                           question, knowledge, gt_answer))

    Qa = compute_alignment_features(Ggt, Gq)
    Ka = compute_alignment_features(Ggt, Gk)

    for k, v in Qa.items(): feats_gt[f"Q_{k}"] = v
    for k, v in Ka.items(): feats_gt[f"K_{k}"] = v

    feats_gt["binary_label"] = 0
    feats_gt["category"] = "none"

    rows.append(feats_gt)
    return rows


############################################################
# MAIN
############################################################

def main():
    print("\n=== PART 2 — FEATURE EXTRACTION ===\n")

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"Loaded MedHallu: {len(ds)} samples")

    # Train/Val/Test split
    idxs = list(range(len(ds)))
    train_idx, tmp = train_test_split(idxs, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(tmp, test_size=0.5, random_state=42)

    # Build map
    split_map = {}
    for i in train_idx: split_map[i] = "train"
    for i in val_idx:   split_map[i] = "validation"
    for i in test_idx:  split_map[i] = "test"

    fout = open(OUTFILE, "w", encoding="utf-8")

    for i, sample in enumerate(ds):
        print("\n" + "-"*80)
        print(f"Processing Sample {i}/{len(ds)}")
        print("-"*80)

        try:
            rows = process_sample(i, sample)
            for r in rows:
                r["split"] = split_map[i]
                fout.write(json.dumps(r) + "\n")
        except Exception as e:
            print(f"⚠ Error on sample {i}: {e}")
            traceback.print_exc()

    fout.close()
    print("\n✓ Features saved to:", OUTFILE)
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()
