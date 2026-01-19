import argparse
import glob
import json
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import file_matrix


def calculate_accuracy_score(C):
    """
    Calculate the accuracy score from a normalized confusion matrix.
    
    Parameters:
    C (np.array): Normalized confusion matrix.
    
    Returns:
    float: Accuracy score (between 0 and 1).
    """
    return np.trace(C)/ C.shape[0]*100

def compute_modularity_louvain(G: nx.DiGraph) -> float:
    """
    Computes the modularity of a graph using the Louvain modularity optimization method.

    Modularity is a measure of the structure of networks or graphs. It quantifies the strength of division 
    of a graph into clusters (communities), where higher values indicate better-defined community structure.

    Args:
        G: nx.DiGraph: A directed graph.
    
    Returns:
        float: The computed modularity value, a measure of the quality of the graph's community structure.

    """
    # Modularity using Louvain modularity communities
    communities= nx.algorithms.community.louvain_communities(G, weight='weight')
    modularity_louv  = nx.algorithms.community.modularity(G, communities, weight='weight')
    return modularity_louv, communities

def community_colors(n_communities: int) -> list:
    """
    Generate a list of colors for the communities, starting with the colors from the paper and continuing with random colors if needed.
    :param n_communities: Number of communities
    :return: List of colors for each community
    """
    # Define a set of colors from the paper
    colors = [
        (1.0, 0.701, 0.0),  # FFB300
        (0.768, 0.690, 1.0),  # C4B0FF
        (0.902, 0.267, 0.686),  # E643AF
        (0.769, 0.553, 0.784),  # C48DC9
        (0.431, 0.459, 1.0),  # 6E75FF
        (0.753, 0.753, 0.753),  # C0C0C0
        (0.239, 0.796, 1.0),  # 3DCBFF
        (0.545, 0.860, 0.573),  # 8BDB92
        (0.91, 0.592, 0.510),  # E89782
    ]

    # Generate random colors if needed
    while len(colors) < n_communities:
        colors.append((np.random.rand(), np.random.rand(), np.random.rand()))

    return colors[:n_communities]

def enforce_minimum_spacing(pos: dict[str, np.ndarray],
                            min_dist: float = 0.1,
                            n_iter: int = 50,
                            decay: float = 0.9) -> dict[str, np.ndarray]:
    """
    Given a dict of positions {node: np.array([x, y])}, iteratively push 
    nodes apart so that every pair is at least min_dist apart.

    Args:
        pos: initial positions (2D) for each node.
        min_dist: smallest allowed distance between any two nodes.
        n_iter: number of relaxation iterations.
        decay: each iteration we multiply the displacement magnitude by decay.

    Returns:
        A new dict of positions with no pair closer than min_dist (approximately).
    """
    # make a working copy
    pos = {n: p.copy() for n, p in pos.items()}
    nodes = list(pos.keys())
    for iteration in range(n_iter):
        max_shift = 0.0
        # for each unordered pair
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                delta = pos[v] - pos[u]
                dist = np.linalg.norm(delta)
                if dist < 1e-6:
                    # if exactly overlapping, nudge randomly
                    delta = np.random.randn(2) * 1e-3
                    dist = np.linalg.norm(delta)
                if dist < min_dist:
                    # how much we need to push them apart
                    shift = (min_dist - dist) / 2.0
                    direction = delta / dist
                    disp = direction * shift * (decay ** iteration)
                    pos[u] -= disp
                    pos[v] += disp
                    max_shift = max(max_shift, np.linalg.norm(disp))
        # if no significant shifts, break early
        if max_shift < 1e-4:
            break
    return pos

def compute_community_layout(G, communities, radius=50.0, seed=42):
    """
    Compute layout with visual separation between communities,
    placing communities in a circular arrangement.
    """
    pos = {}
    positioned_nodes = set()
    num_communities = len(communities)

    # Arrange community centers in a circle
    angle_step = 2 * np.pi / max(num_communities, 1)
    radius = 30* num_communities

    for idx, community in enumerate(communities):
        subG = G.subgraph(community)

        local_pos = nx.kamada_kawai_layout(subG, weight='weight', scale=10)
        local_pos = enforce_minimum_spacing(local_pos, 6* num_communities)

        # Compute center position for this community
        angle = idx * angle_step
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)

        # Offset all positions in the sublayout
        for node in local_pos:
            local_pos[node][0] += center_x
            local_pos[node][1] += center_y

        pos.update(local_pos)
        positioned_nodes.update(community)

    # Handle nodes not in any community
    remaining_nodes = set(G.nodes()) - positioned_nodes
    if remaining_nodes:
        remaining_subG = G.subgraph(remaining_nodes)
        remaining_pos = nx.kamada_kawai_layout(remaining_subG, weight='weight', scale=10)
        # Place those in the center
        for node in remaining_pos:
            remaining_pos[node][0] += 0
            remaining_pos[node][1] += 0
        pos.update(remaining_pos)

    return pos

def average_color(c1, c2):
    return tuple((np.array(c1) + np.array(c2)) / 2)

def plot_graph(G: nx.DiGraph, filepath: str, communities: list[list[int]]=None, colors_communities: list[list[float]]=None, title: str=None, node_names: list[str]=None):
    """Plot a picture of a graph and store it a defined place.
    Args:
        G: Graph to be plotted.
        filepath: path to file where the result should be written
        communities (optional): indicated the identified communities by color. Leave None for detection of communities by modularity.
        colors_communities (optional): list of colors for each community, if None, random colors will be used.
        title (optional): title of the graph
    """

    node_colors = {}
    if communities is None:
        modularity_louvain, communities = compute_modularity_louvain(G)


    for i, community in enumerate(communities):
        color = [np.random.random() for _ in range(3)] if colors_communities is None else colors_communities[i]
        for node in community:
            node_colors[node] = color

    pos = compute_community_layout(G, communities)

    if node_names is not None:
        relabel_map = {old: new for old, new in zip(G.nodes(), node_names)}
        G = nx.relabel_nodes(G, relabel_map, copy=True)
        pos = {relabel_map[k]: v for k, v in pos.items()}
        node_colors = {relabel_map[k]: v for k, v in node_colors.items()}

    node_color_array = [node_colors[node] for node in G.nodes()]


    # Plot the graph
    plt.figure(figsize=(32, 24))
    plt.axis('off')  # Turn off the axis
    nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color=node_color_array)
    nx.draw_networkx_labels(G, pos=pos, font_size=10, font_color='black')
    
    # Collect all edges and their weights
    edges = list(G.edges(data=True))
    edge_weights = [data.get('weight', 1.0) for _, _, data in edges]  # default weight = 1.0
    scale_factor = 15.0
    boost_factor = 0.3
    

    # Apply exponential scaling
    edge_widths = np.exp(scale_factor * np.array(edge_weights))

    # Boost everything uniformly
    edge_widths *= boost_factor
    edge_widths = np.clip(edge_widths, None, 7)

    edge_colors = [
        average_color(node_colors[u], node_colors[v]) 
        for u, v in G.edges()
    ]

    nx.draw_networkx_edges(G, pos=pos, edgelist=[(u, v) for u, v, _ in edges], width=edge_widths, edge_color=edge_colors, alpha=0.7, arrows=True, connectionstyle="arc3,rad=0.2", arrowsize = 15)

    
    if title is not None:
        plt.title(title)
    
    # Save the plot as a PDF file
    plt.savefig(filepath)

   

# Fine labels as provided in the Tiny ImageNet dataset
fine_labels = [
 'goldfish', # id 0
'European fire salamander', # id 1
'bullfrog', # id 2
'tailed frog', # id 3
'American alligator', # id 4
'boa constrictor', # id 5
'trilobite', # id 6
'scorpion', # id 7
'black widow', # id 8
'tarantula', # id 9
'centipede', # id 10
'goose', # id 11
'koala', # id 12
'jellyfish', # id 13
'brain coral', # id 14
'snail', # id 15
'slug', # id 16
'sea slug', # id 17
'American lobster', # id 18
'spiny lobster', # id 19
'black stork', # id 20
'king penguin', # id 21
'albatross', # id 22
'dugong', # id 23
'Chihuahua', # id 24
'Yorkshire terrier', # id 25
'golden retriever', # id 26
'Labrador retriever', # id 27
'German shepherd', # id 28
'standard poodle', # id 29
'tabby', # id 30
'Persian cat', # id 31
'Egyptian cat', # id 32
'cougar', # id 33
'lion', # id 34
'brown bear', # id 35
'ladybug', # id 36
'fly', # id 37
'bee', # id 38
'grasshopper', # id 39
'walking stick', # id 40
'cockroach', # id 41
'mantis', # id 42
'dragonfly', # id 43
'monarch', # id 44
'sulphur butterfly', # id 45
'sea cucumber', # id 46
'guinea pig', # id 47
'hog', # id 48
'ox', # id 49
'bison', # id 50
'bighorn', # id 51
'gazelle', # id 52
'Arabian camel', # id 53
'orangutan', # id 54
'chimpanzee', # id 55
'baboon', # id 56
'African elephant', # id 57
'lesser panda', # id 58
'abacus', # id 59
'academic gown', # id 60
'altar', # id 61
'apron', # id 62
'backpack', # id 63
'bannister', # id 64
'barbershop', # id 65
'barn', # id 66
'barrel', # id 67
'basketball', # id 68
'bathtub', # id 69
'beach wagon', # id 70
'beacon', # id 71
'beaker', # id 72
'beer bottle', # id 73
'bikini', # id 74
'binoculars', # id 75
'birdhouse', # id 76
'bow tie', # id 77
'brass', # id 78
'broom', # id 79
'bucket', # id 80
'bullet train', # id 81
'butcher shop', # id 82
'candle', # id 83
'cannon', # id 84
'cardigan', # id 85
'cash machine', # id 86
'CD player', # id 87
'chain', # id 88
'chest', # id 89
'Christmas stocking', # id 90
'cliff dwelling', # id 91
'computer keyboard', # id 92
'confectionery', # id 93
'convertible', # id 94
'crane', # id 95
'dam', # id 96
'desk', # id 97
'dining table', # id 98
'drumstick', # id 99
'dumbbell', # id 100
'flagpole', # id 101
'fountain', # id 102
'freight car', # id 103
'frying pan', # id 104
'fur coat', # id 105
'gasmask', # id 106
'go-kart', # id 107
'gondola', # id 108
'hourglass', # id 109
'iPod', # id 110
'jinrikisha', # id 111
'kimono', # id 112
'lampshade', # id 113
'lawn mower', # id 114
'lifeboat', # id 115
'limousine', # id 116
'magnetic compass', # id 117
'maypole', # id 118
'military uniform', # id 119
'miniskirt', # id 120
'moving van', # id 121
'nail', # id 122
'neck brace', # id 123
'obelisk', # id 124
'oboe', # id 125
'organ', # id 126
'parking meter', # id 127
'pay-phone', # id 128
'picket fence', # id 129
'pill bottle', # id 130
'plunger', # id 131
'pole', # id 132
'police van', # id 133
'poncho', # id 134
'pop bottle', # id 135
'potters wheel', # id 136
'projectile', # id 137
'punching bag', # id 138
'reel', # id 139
'refrigerator', # id 140
'remote control', # id 141
'rocking chair', # id 142
'rugby ball', # id 143
'sandal', # id 144
'school bus', # id 145
'scoreboard', # id 146
'sewing machine', # id 147
'snorkel', # id 148
'sock', # id 149
'sombrero', # id 150
'space heater', # id 151
'spider web', # id 152
'sports car', # id 153
'steel arch bridge', # id 154
'stopwatch', # id 155
'sunglasses', # id 156
'suspension bridge', # id 157
'swimming trunks', # id 158
'syringe', # id 159
'teapot', # id 160
'teddy', # id 161
'thatch', # id 162
'torch', # id 163
'tractor', # id 164
'triumphal arch', # id 165
'trolleybus', # id 166
'turnstile', # id 167
'umbrella', # id 168
'vestment', # id 169
'viaduct', # id 170
'volleyball', # id 171
'water jug', # id 172
'water tower', # id 173
'wok', # id 174
'wooden spoon', # id 175
'comic book', # id 176
'plate', # id 177
'guacamole', # id 178
'ice cream', # id 179
'ice lolly', # id 180
'pretzel', # id 181
'mashed potato', # id 182
'cauliflower', # id 183
'bell pepper', # id 184
'mushroom', # id 185
'orange', # id 186
'lemon', # id 187
'banana', # id 188
'pomegranate', # id 189
'meat loaf', # id 190
'pizza', # id 191
'potpie', # id 192
'espresso', # id 193
'alp', # id 194
'cliff', # id 195
'coral reef', # id 196
'lakeside', # id 197
'seashore', # id 198
'acorn', # id 199
]

# Mapping fine labels to indices
label_to_index = {label: idx for idx, label in enumerate(fine_labels)}


# Main Script  
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate target metrics for confusion matrices.')
    parser.add_argument('--matrix-dir', type=str, required=True,
                        help='Directory containing the confusion matrix files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output metrics/plots.')                      
    parser.add_argument('--alpha', type=float, help='The parameter determining the cross entropy mix between true and predicted labels.', default=1.0)
    parser.add_argument('--train-set', action='store_true', help='Use training set confusion matrices.')
    parser.add_argument('--plot-metrics', action='store_true', help='Plot metrics after computation.')
    parser.add_argument('--plot-graphs', action='store_true', help='Plot graphs according to modularity communities.')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"alpha_{args.alpha}")
    os.makedirs(output_dir, exist_ok=True)

    matrix_dir = os.path.join(args.matrix_dir, f"alpha_{args.alpha}")
    matrix_files = glob.glob(f"{matrix_dir}/*.npy")

    matrix_files = glob.glob(f"{matrix_dir}/*.npy")
    if not matrix_files:
        raise FileNotFoundError(f"No confusion matrix files found in {matrix_dir}")

    metrics = {}

    fm, epochs, iterations, layers = file_matrix(matrix_files)
    print(f"Found {len(epochs)} epochs, {len(iterations)} iterations, and {len(layers)} layers.")

    split_index = 0 if args.train_set else 1

    if args.plot_graphs:
        graph_dir = os.path.join(args.output_dir, f"alpha_{args.alpha}", "graphs")
        colors = community_colors(200)

    # Iterate through epochs, iterations, and layers
    for (i, epoch) in enumerate(epochs):
        metrics[epoch] = {}
        for (j, layer) in enumerate(layers):
            metrics[epoch][layer] = {}
            for (k, iteration) in enumerate(iterations):
                metrics[epoch][layer][iteration] = {}
                # Get the confusion matrix file for the current epoch, layer, and iteration
                matrix_file = fm[i, k, j, split_index]  # Use test set confusion matrix

                C = np.load(matrix_file)
                if C.ndim != 2 or C.shape[0] != C.shape[1]:
                    raise ValueError(f"Invalid confusion matrix shape in file {matrix_file}: {C.shape}")

                C_directed = C - np.diag(np.diag(C))

                # Create graph from the matrix.
                G_directed = nx.from_numpy_array(C_directed, create_using=nx.DiGraph())

                modularity_louvain, communities_louvain = compute_modularity_louvain(G_directed)
                accuracy = calculate_accuracy_score(C)

                if args.plot_graphs:
                    file_path = os.path.join(graph_dir, f'epoch_{epochs[i]}_iteration_{iterations[k]}_layer_{layers[j]}.svg')
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    plot_graph(G_directed, filepath=file_path, communities=communities_louvain, colors_communities=colors,
                    title=f'Epoch {epochs[i]} Iteration {iterations[k]} Layer {layers[j]}', node_names=fine_labels)



                metrics[epoch][layer][iteration] = {
                    "modularity_louvain": float(modularity_louvain),
                    "nr_communities_louvain": int(len(communities_louvain)),
                    "accuracy": float(accuracy)
                }


                print("=== Graph Analysis Metrics ===")
                print(f"Epoch: {epoch}, Layer: {layer}, Iteration: {iteration}")

                # Modularity (Louvain)
                print(f"Modularity (Louvain): {modularity_louvain:.4f}")
                print(f"Number of Louvain Communities: {len(communities_louvain)}\n")

                # Accuracy
                print(f"Accuracy: {accuracy:.4f}")


    # Save metrics to json file
    output_file = f"{output_dir}/graph_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if args.plot_metrics:
        # Plot the metrics for all layers averaged over iterations across epochs
        # Modularity (Louvain)
        modularity_values = np.array([[[metrics[epoch][layer][iteration]["modularity_louvain"]
                                       for iteration in iterations] for layer in layers] for epoch in epochs])
        avg_modularity = np.mean(modularity_values, axis=2)
        plt.figure(figsize=(10, 6))
        for i, layer in enumerate(layers):
            plt.plot(epochs, avg_modularity[:, i], label=f'Layer {layer}')
        plt.title('Average Modularity (Louvain) Across Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Modularity (Louvain)')
        plt.legend(loc='upper left')
        plt.savefig(f"{output_dir}/modularity_louvain.png")

        