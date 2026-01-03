#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PART 1 — LATENT GRAPH BUILDER (UPDATED VERSION)

This upgraded module implements the full pipeline described in the methodology:

 ✔ Multi-agent LLM extraction (Gemma 3 12B via Ollama)
 ✔ Free-form entity extraction (no ontology)
 ✔ Free-form relation extraction
 ✔ Embedding-based entity normalization across Q/E/A
 ✔ Construction of latent semantic graphs for:
      - Question (Gq)
      - Evidence (Gk)
      - Answer   (Ga)
 ✔ Vector embeddings for nodes + edges
 ✔ JSON serialization for Part 2 (feature extraction)

This version is fully aligned with the final hallucination detection approach.
"""

import os, json, time, traceback, regex as re
import numpy as np
import networkx as nx
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from ollama import chat as ollama_chat


###########################################################
# CONFIG
###########################################################


OLLAMA_MODEL = "gemma3:12b"
DATASET_NAME = "UTAustin-AIHealth/MedHallu"
DATASET_CONFIG = "pqa_artificial"
DATASET_SPLIT = "train"

MAX_SAMPLES = 9000  # adjust as needed

CACHE_DIR = "llm_cache_unlabeld"
OUTPUT_DIR = "graph_outputs_not_labled"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding model for nodes + relations + normalization
EMBEDDER = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
EMB_DIM = EMBEDDER.get_sentence_embedding_dimension()


###########################################################
# UTILS — TEXT EMBEDDINGS
###########################################################

def embed(text: str) -> np.ndarray:
    """Return dense vector embedding for a text."""
    return EMBEDDER.encode(text, convert_to_numpy=True)


###########################################################
# ROBUST JSON CLEANER
###########################################################

def extract_json(text):
    """Attempts to extract JSON from raw LLM output."""
    if not text:
        return None

    # Try direct
    try:
        return json.loads(text)
    except:
        pass

    # Strip ```json fences
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except:
        pass

    # Regex fallback
    match = re.search(r'\{(?:[^{}]|(?R))*\}', text, flags=re.S)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


###########################################################
# SAFE OLLAMA CALL
###########################################################

def llm_call(system_prompt, user_prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = ollama_chat(
                OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            if "message" in resp:
                return resp["message"]["content"]
            return None

        except Exception as e:
            print(f"[LLM ERROR attempt {attempt+1}] {e}")
            time.sleep(2)

    return None


###########################################################
# CACHING DECORATOR
###########################################################

def cached(func):
    """Caches LLM results to avoid recomputation."""
    def wrapper(arg1, arg2=None):
        key = f"{func.__name__}-{hash(str(arg1)+str(arg2))}.json"
        path = os.path.join(CACHE_DIR, key)

        if os.path.exists(path):
            try:
                return json.load(open(path, "r", encoding="utf-8"))
            except:
                pass

        result = func(arg1, arg2) if arg2 else func(arg1)
        json.dump(result, open(path, "w", encoding="utf-8"))
        return result

    return wrapper


###########################################################
# LLM PROMPTS (FREE-FORM GRAPH EXTRACTION)
###########################################################

ENTITY_PROMPT = """
You are a biomedical concept extractor.

Extract all meaningful biomedical entities from the text.
These can include diseases, drugs, symptoms, proteins, risk factors,
biological processes, or any concept relevant to biomedical reasoning.

Use a free-form approach. DO NOT assign predefined types.

Return ONLY valid JSON:
{
  "entities": [
    {"id":"E1", "text":"..."},
    ...
  ]
}
"""

RELATION_PROMPT = """
Extract relations between the given entities within the text.

Relations must be short natural language phrases (2–6 words).
Use ONLY what the text explicitly supports.

Return ONLY valid JSON:
{
  "relations": [
    {"head":"E1", "relation":"...", "tail":"E2"},
    ...
  ]
}
"""


###########################################################
# AGENT FUNCTIONS
###########################################################

@cached
def extract_entities(text):
    """Entity extraction agent."""
    raw = llm_call(
        ENTITY_PROMPT,
        f"Extract entities.\nText:\n{text}\nReturn JSON only."
    )
    js = extract_json(raw)
    if not js or "entities" not in js:
        return {"entities": []}
    return js


@cached
def extract_relations(text, entities):
    """Relation extraction agent."""
    if not entities or "entities" not in entities:
        return {"relations": []}

    raw = llm_call(
        RELATION_PROMPT,
        f"Text:\n{text}\n\nEntities:\n{json.dumps(entities)}\nReturn JSON only."
    )
    js = extract_json(raw)
    if not js or "relations" not in js:
        return {"relations": []}
    return js


###########################################################
# ENTITY NORMALIZATION (EMBEDDING-BASED)
###########################################################

def normalize_entities(all_entity_lists, threshold=0.80):
    """
    Given lists of entities from Q/E/A, merge them by embedding similarity.

    Input:
        all_entity_lists = [ents_q, ents_k, ents_a]
        where each is {"entities": [{"id":...,"text":...}, ...]}

    Output:
        global_entities: [{"gid": "G1", "text": "...", "vec": [...]}]
        mappings: per-graph mapping from local entity ids → global ids
    """

    # Step 1: flatten all
    merged = []
    for source_idx, entlist in enumerate(all_entity_lists):
        for ent in entlist["entities"]:
            vec = embed(ent["text"])
            merged.append({
                "source": source_idx,  # 0=Q,1=K,2=A
                "eid": ent["id"],
                "text": ent["text"],
                "vec": vec
            })

    # Step 2: cluster by similarity
    global_groups = []
    used = [False] * len(merged)

    for i, item in enumerate(merged):
        if used[i]:
            continue

        group = [i]
        used[i] = True

        for j in range(i+1, len(merged)):
            if used[j]:
                continue
            sim = np.dot(item["vec"], merged[j]["vec"]) / (
                np.linalg.norm(item["vec"]) * np.linalg.norm(merged[j]["vec"]) + 1e-9
            )
            if sim >= threshold:
                used[j] = True
                group.append(j)

        global_groups.append(group)

    # Step 3: construct global entities
    global_entities = []
    gid_map = {}
    mappings = {0:{}, 1:{}, 2:{}}

    for idx, group in enumerate(global_groups):
        gid = f"G{idx+1}"
        texts = [merged[g]["text"] for g in group]
        centroid = np.mean([merged[g]["vec"] for g in group], axis=0)

        global_entities.append({
            "gid": gid,
            "examples": texts,
            "text": texts[0],
            "vec": centroid.tolist()
        })

        # Map local → global
        for g in group:
            src = merged[g]["source"]
            local_id = merged[g]["eid"]
            mappings[src][local_id] = gid

    return global_entities, mappings


###########################################################
# GRAPH CONSTRUCTION
###########################################################

def build_graph(text, global_entities, local_entities, local_relations, local_to_global):
    """
    Build a graph (Gq, Gk, or Ga) using normalized entity ids.
    """
    G = nx.DiGraph()

    # --- Add nodes ---
    for ent in local_entities["entities"]:
        if ent["id"] in local_to_global:
            gid = local_to_global[ent["id"]]
            ge = next(x for x in global_entities if x["gid"] == gid)
            G.add_node(gid, text=ge["text"], vec=ge["vec"])

    # --- Add edges ---
    for r in local_relations["relations"]:
        h, rel, t = r["head"], r["relation"], r["tail"]
        if h in local_to_global and t in local_to_global:
            gh = local_to_global[h]
            gt = local_to_global[t]
            etext = f"{G.nodes[gh]['text']} [{rel}] {G.nodes[gt]['text']}"
            G.add_edge(
                gh, gt,
                relation=rel,
                text=etext,
                vec=embed(etext).tolist()
            )

    return G


###########################################################
# SERIALIZATION
###########################################################

def graph_to_json(G):
    data = {"nodes": [], "edges": []}

    for n in G.nodes():
        data["nodes"].append({
            "id": n,
            "text": G.nodes[n]["text"],
            "vec": G.nodes[n]["vec"]
        })

    for u, v in G.edges():
        data["edges"].append({
            "head": u,
            "tail": v,
            "relation": G.edges[u, v]["relation"],
            "text": G.edges[u, v]["text"],
            "vec": G.edges[u, v]["vec"]
        })

    return data


###########################################################
# MAIN SAMPLE PROCESSING
###########################################################

def process_sample(idx, sample):
    print(f"\n▶ Sample {idx} — extracting Q/K/A graphs")

    q = sample["Question"]
    k = " ".join(sample["Knowledge"])
    a = sample["Hallucinated Answer"]
    label = sample["Category of Hallucination"]

    # --- Extract raw entities & relations ---
    ents_q = extract_entities(q)
    ents_k = extract_entities(k)
    ents_a = extract_entities(a)

    rel_q = extract_relations(q, ents_q)
    rel_k = extract_relations(k, ents_k)
    rel_a = extract_relations(a, ents_a)

    # --- Normalize entities across Q/E/A ---
    global_entities, mappings = normalize_entities([ents_q, ents_k, ents_a])

    # --- Build graphs ---
    Gq = build_graph(q, global_entities, ents_q, rel_q, mappings[0])
    Gk = build_graph(k, global_entities, ents_k, rel_k, mappings[1])
    Ga = build_graph(a, global_entities, ents_a, rel_a, mappings[2])

    # --- Serialize ---
    out = {
        "index": idx,
        "label": label,
        "question": q,
        "knowledge": k,
        "answer": a,
        "global_entities": global_entities,
        "graph_question": graph_to_json(Gq),
        "graph_knowledge": graph_to_json(Gk),
        "graph_answer": graph_to_json(Ga)
    }

    path = os.path.join(OUTPUT_DIR, f"sample_{idx:04d}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)

    print(f"   ✔ Saved → {path}")


###########################################################
# MAIN
###########################################################

def main():
    print("Loading MedHallu dataset...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)[DATASET_SPLIT]

    n = min(MAX_SAMPLES, len(ds))
    print(f"Will process {n} samples.")

    for i in range(n):
        try:
            process_sample(i, ds[i])
        except Exception as e:
            print(f"⚠ Error on sample {i}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
