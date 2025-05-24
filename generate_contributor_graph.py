"""
Generate a graph a communities. A community is made of contributors who
share common concerns.
We start by creating of graph of contributors. Communities are found
by partioning the graph of contributors with Louvain algorithm.
Finally, communities are represented as nodes in a graph, with edges
representing the similarity of the communities' concerns.

"""
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from datasets import load_dataset
from pyvis.network import Network
from scipy.spatial.distance import jensenshannon

rng = np.random.default_rng(seed=1234)

# We only select a subset of columns and samples to avoid OOM issues
relevant_cols = ["contribution_id", "cluster_id", "cluster_title"]
N_SAMPLES = 45000

ds = load_dataset("perspectiva-solution/embeddings-gdn-question-163")['train']
df = ds.to_pandas()
df = df.loc[df.cluster_id != -1, relevant_cols].sample(n=N_SAMPLES, random_state=1234) # Remove ideas which were not clustered
n_clusters = df.cluster_title.nunique()

# TODO Probably refactor these functions as methods
# of an object since we keep passing personas around

def sim_from_dist(vec_a, vec_b, dist=jensenshannon):
    return 1 / (1 + dist(vec_a, vec_b))

def characterize_node(idx, personas, id_col="cluster_title"):
    """Return common contributions of the community sorted by frequency of occurence
    """
    msk = df.contribution_id.isin(personas[idx])
    community_demands = df[msk].groupby(id_col)["cluster_title"].count().rename("cluster_count")
    return (community_demands / community_demands.sum()).sort_values(ascending=False)

    
def node_to_vec(node, personas):
    char = characterize_node(node, personas, id_col="cluster_id")
    vec = np.zeros((n_clusters))
    for _, row in pd.DataFrame(char).reset_index().iterrows():
        vec[int(row.cluster_id)] = row.cluster_count
    return vec

def make_title(demands, topk=5):
    title = ""
    for row_idx in range(topk):
        try:
            row_value = demands.iloc[row_idx]
        except IndexError:
            return title
        demand = demands.index[row_idx]
        title += f"{demand} ({row_value * 100:.2f}%)\n"
    return title


def find_bridge(node_pair, personas, topk=5):
    common_ideas = pd.merge(
        characterize_node(node_pair[0], personas)[:topk],
        characterize_node(node_pair[1], personas)[:topk],
        how="inner",
        on="cluster_title"
    ).reset_index()
    common_ideas["sum_freq"] = common_ideas["cluster_count_x"] + common_ideas["cluster_count_y"]
    bridge = common_ideas.sort_values(by="sum_freq", ascending=False).loc[0, "cluster_title"]
    return bridge


# Create contributors graph
G = nx.Graph()
G.add_nodes_from(df.contribution_id) # A node is a contribution

# Add edge if two nodes belong to the same cluster
for title, subdf in df.groupby("cluster_title"):
    edges = tuple(combinations(subdf.contribution_id.to_list(), 2))
    for edge in edges:
        if G.has_edge(*edge):
            G.edges[*edge]["weight"] += 1
        else:
            G.add_edge(*edge, weight=1)

# Find a partition of the graph
# Personas are defined as "big enough" (>threshold) communities
partition = nx.community.louvain_communities(
    G, weight="weight", resolution=1, seed=2025
)
personas = []
threshold = 10
for node_set in partition:
    if len(node_set) >= threshold:
        personas.append(node_set)


# Set scaling factors for edge and node display
EDGE_SCALE_FACTOR = 0.1
max_comm_size = max((len(p) for p in personas))
size_factor = 100 / max_comm_size

# Create graph of Louvain communities
H = nx.Graph()
H.add_nodes_from(range(len(personas)), borderWidth=0.5)
n_tests = len(tuple(combinations(H.nodes, 2)))

weights = []
for node_pair in combinations(H.nodes, 2):
    vec_a, vec_b = node_to_vec(node_pair[0], personas), node_to_vec(node_pair[1], personas)

    try:
        bridge = find_bridge(node_pair, personas)
    except KeyError: # If there's not a frequent common idea, skip edge
        continue

    sim = sim_from_dist(vec_a, vec_b, dist=jensenshannon)
    weights.append(sim)
    if sim < 0.5:
        color = "gray"
    else:
        color = "black"

    H.add_edge(
        *node_pair,
        value=sim * EDGE_SCALE_FACTOR,
        hidden=False,
        color=color,
        title=bridge
    )


for community in H.nodes:
    community_demands = characterize_node(community, personas)
    H.nodes[community]["title"] = make_title(community_demands)
    H.nodes[community]["size"] = len(personas[community]) * size_factor

# Plot graph of communities with Pyviz
nt = Network("500px", "1000px")
nt.force_atlas_2based()
nt.from_nx(H)
nt.toggle_physics(True)
nt.show_buttons(filter_=["physics"])


nt.options["interaction"].__dict__["selectConnectedEdges"] = True

nt.show(f"communities_ncontrib-{N_SAMPLES}_communitythresh-{threshold}.html", notebook=False)


