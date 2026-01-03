from _1_graph_builder import extract_entities, extract_relations, embed
import networkx as nx
import numpy as np

def build_graph(text):
    ents = extract_entities(text)
    rels = extract_relations(text, ents)

    G = nx.DiGraph()

    for e in ents["entities"]:
        vec = embed(e["text"])
        G.add_node(e["id"], text=e["text"], vec=vec.tolist())

    for r in rels["relations"]:
        if r["head"] in G.nodes and r["tail"] in G.nodes:
            etext = f"{G.nodes[r['head']]['text']} [{r['relation']}] {G.nodes[r['tail']]['text']}"
            G.add_edge(
                r["head"], r["tail"],
                relation=r["relation"],
                text=etext,
                vec=embed(etext).tolist()
            )

    return G
