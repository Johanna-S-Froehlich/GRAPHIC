import argparse
import glob
import json
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import file_matrix


def create_fine_labels_for_coarse_categories(mapping_superclass_fine, mapping_make_superclass):
    """
    Creates a new dictionary where the keys are 'natural' and 'man-made',
    and the values are the fine labels corresponding to the superclasses in those categories.

    :param mapping_superclass_fine: Dictionary mapping superclasses to fine labels.
    :param mapping_make_superclass: Dictionary mapping coarse categories to superclasses.
    :return: New dictionary with 'natural' and 'man-made' as keys, and fine labels as values.
    """
    fine_labels_for_coarse = {}

    # Iterate over the coarse categories ('natural', 'man-made')
    for coarse_group, superclasses in mapping_make_superclass.items():
        # Initialize the list to hold fine labels for the current coarse category
        fine_labels_for_coarse[coarse_group] = []

        # For each superclass in the list of superclasses for this coarse category
        for superclass in superclasses:
            # Retrieve fine labels for the current superclass from mapping_superclass_fine
            if superclass in mapping_superclass_fine:
                fine_labels_for_coarse[coarse_group].extend(mapping_superclass_fine[superclass])

    return fine_labels_for_coarse

def calculate_association_matrix(C, fine_labels, fine_to_group_dict):
    """
    Computes the association matrix E from the confusion matrix C and class-to-group mapping,
    normalized by group sizes.
    :param C: Confusion matrix (weighted adjacency matrix).
    :param fine_labels: List of fine labels corresponding to rows/columns of C.
    :param fine_to_group_dict: Dictionary mapping group names to lists of fine labels.
    :return: Normalized association matrix E.
    """

    # Map fine labels to their respective group indices
    fine_to_group = {}
    for group_idx, (group, fine_list) in enumerate(fine_to_group_dict.items()):
        for fine_label in fine_list:
            fine_to_group[fine_label] = group_idx

    # Compute group sizes
    group_sizes = {group_idx: len(fine_list) for group_idx, fine_list in enumerate(fine_to_group_dict.values())}
    
    # Number of groups
    n_superclasses = len(fine_to_group_dict)

    # Initialize the association matrix E
    E = np.zeros((n_superclasses, n_superclasses))

    # Iterate over all pairs of fine labels in the confusion matrix
    for i in range(len(C)):
        for j in range(len(C)):
            fine_label_i = fine_labels[i]
            fine_label_j = fine_labels[j]
            
            # Get the group indices for the fine labels
            group_i = fine_to_group[fine_label_i]
            group_j = fine_to_group[fine_label_j]
            
            # Normalize by group sizes
            normalization_factor = group_sizes[group_i] * group_sizes[group_j]
            if normalization_factor > 0:
                E[group_i, group_j] += C[i, j] / normalization_factor

    # Normalize E to represent fractions of total edge weights
    E /= np.sum(E)

    return E

def calculate_accuracy_score(C):
    """
    Calculate the accuracy score from a normalized confusion matrix.
    
    Parameters:
    C (np.array): Normalized confusion matrix.
    
    Returns:
    float: Accuracy score (between 0 and 1).
    """
    return np.trace(C)/ C.shape[0]*100

def compute_weighted_assortativity(C, fine_labels, class_to_superclass):
    """
    Computes the weighted assortativity coefficient using Newman's formula.
    :param C: Confusion matrix (weighted adjacency matrix).
    :param class_to_superclass: Dictionary mapping fine labels to superclasses.
    :param n_superclasses: Total number of superclasses.
    :return: Weighted assortativity coefficient.
    """
    E = calculate_association_matrix(C, fine_labels, class_to_superclass)
    r = (np.sum(np.diag(E))-np.linalg.norm(E)**2)/(1-np.linalg.norm(E)**2)
    return r

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
    plt.figure(figsize=(16, 12))
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

# Fine labels as provided in the CIFAR-100 dataset
fine_labels = [
    'apple',              # id 0
    'aquarium_fish',      # id 1
    'baby',               # id 2
    'bear',               # id 3
    'beaver',             # id 4
    'bed',                # id 5
    'bee',                # id 6
    'beetle',             # id 7
    'bicycle',            # id 8
    'bottle',             # id 9
    'bowl',               # id 10
    'boy',                # id 11
    'bridge',             # id 12
    'bus',                # id 13
    'butterfly',          # id 14
    'camel',              # id 15
    'can',                # id 16
    'castle',             # id 17
    'caterpillar',        # id 18
    'cattle',             # id 19
    'chair',              # id 20
    'chimpanzee',         # id 21
    'clock',              # id 22
    'cloud',              # id 23
    'cockroach',          # id 24
    'couch',              # id 25
    'crab',               # id 26
    'crocodile',          # id 27
    'cup',                # id 28
    'dinosaur',           # id 29
    'dolphin',            # id 30
    'elephant',           # id 31
    'flatfish',           # id 32
    'forest',             # id 33
    'fox',                # id 34
    'girl',               # id 35
    'hamster',            # id 36
    'house',              # id 37
    'kangaroo',           # id 38
    'computer_keyboard',  # id 39
    'lamp',               # id 40
    'lawn_mower',         # id 41
    'leopard',            # id 42
    'lion',               # id 43
    'lizard',             # id 44
    'lobster',            # id 45
    'man',                # id 46
    'maple_tree',         # id 47
    'motorcycle',         # id 48
    'mountain',           # id 49
    'mouse',              # id 50
    'mushroom',           # id 51
    'oak_tree',           # id 52
    'orange',             # id 53
    'orchid',             # id 54
    'otter',              # id 55
    'palm_tree',          # id 56
    'pear',               # id 57
    'pickup_truck',       # id 58
    'pine_tree',          # id 59
    'plain',              # id 60
    'plate',              # id 61
    'poppy',              # id 62
    'porcupine',          # id 63
    'possum',             # id 64
    'rabbit',             # id 65
    'raccoon',            # id 66
    'ray',                # id 67
    'road',               # id 68
    'rocket',             # id 69
    'rose',               # id 70
    'sea',                # id 71
    'seal',               # id 72
    'shark',              # id 73
    'shrew',              # id 74
    'skunk',              # id 75
    'skyscraper',         # id 76
    'snail',              # id 77
    'snake',              # id 78
    'spider',             # id 79
    'squirrel',           # id 80
    'streetcar',          # id 81
    'sunflower',          # id 82
    'sweet_pepper',       # id 83
    'table',              # id 84
    'tank',               # id 85
    'telephone',          # id 86
    'television',         # id 87
    'tiger',              # id 88
    'tractor',            # id 89
    'train',              # id 90
    'trout',              # id 91
    'tulip',              # id 92
    'turtle',             # id 93
    'wardrobe',           # id 94
    'whale',              # id 95
    'willow_tree',        # id 96
    'wolf',               # id 97
    'woman',              # id 98
    'worm',               # id 99
]

# Mapping from superclasses to fine labels
mapping_superclass_fine = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],                                     # 0
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],                                         # 1
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],                                           # 2
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],                                           # 3
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],                        # 4
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],       # 5
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],                                  # 6
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],                                  # 7
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],                                       # 8
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],                   # 9
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],                        # 10
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],            # 11
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],                             # 12
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],                             # 13
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],                                                      # 14
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],                                     # 15
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],                                   # 16
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],                           # 17
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],                                # 18
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],                                 # 19
}

# Mapping from make to superclasses
mapping_make_superclass = {
    'natural': ['aquatic mammals', 'fish', 'flowers', 'fruit and vegetables', 'insects', 'large carnivores',                
    'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
    'people', 'reptiles', 'small mammals', 'trees'],                                                                        
    'man-made': ['food containers', 'household electrical device', 'household furniture', 'large man-made outdoor things',  
    'vehicles 1', 'vehicles 2'],                                                                                            
}

mapping_intuitive_fine = {
    'animal on land': [ 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'snail', 'spider', 'worm', 'dinosaur', 'lizard', 'snake', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'animal in water': ['crab', 'lobster', 'beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout', 'crocodile', 'turtle'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'home stuff': ['bottle', 'bowl', 'can', 'cup', 'plate', 'clock', 'computer_keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'outside structures': ['cloud', 'forest', 'mountain', 'plain', 'sea',  'bridge', 'castle', 'house', 'road', 'skyscraper'],
    'plants': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
}

# Mapping fine labels to indices
label_to_index = {label: idx for idx, label in enumerate(fine_labels)}

# Replace class names with indices
superclass_communities = [
    [label_to_index[fine_label] for fine_label in fine_list]
    for fine_list in mapping_superclass_fine.values()
]

mappings = {
"superclass_fine": mapping_superclass_fine,
"intuitive_fine": mapping_intuitive_fine,
"make_superclass": create_fine_labels_for_coarse_categories(mapping_superclass_fine, mapping_make_superclass),
}

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
        colors = community_colors(100)

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
                assort_nat_man = compute_weighted_assortativity(C_directed, fine_labels, mappings["make_superclass"])
                assort_super = compute_weighted_assortativity(C_directed, fine_labels, mappings["superclass_fine"])
                assort_intuitive = compute_weighted_assortativity(C_directed, fine_labels, mappings["intuitive_fine"])
                accuracy = calculate_accuracy_score(C)

                if args.plot_graphs:
                    file_path = os.path.join(graph_dir, f'epoch_{epochs[i]}_iteration_{iterations[k]}_layer_{layers[j]}.svg')
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    plot_graph(G_directed, filepath=file_path, communities=communities_louvain, colors_communities=colors,
                    title=f'Epoch {epochs[i]} Iteration {iterations[k]} Layer {layers[j]}', node_names=fine_labels)



                metrics[epoch][layer][iteration] = {
                    "modularity_louvain": float(modularity_louvain),
                    "nr_communities_louvain": int(len(communities_louvain)),
                    "assort_nat_man": float(assort_nat_man),
                    "assort_super": float(assort_super),
                    "assort_intuitive": float(assort_intuitive),
                    "accuracy": float(accuracy)
                }

                print("=== Graph Analysis Metrics ===")
                print(f"Epoch: {epoch}, Layer: {layer}, Iteration: {iteration}")

                # Modularity (Louvain)
                print(f"Modularity (Louvain): {modularity_louvain:.4f}")
                print(f"Number of Louvain Communities: {len(communities_louvain)}\n")

                # Assortativity Coefficients
                print("Assortativity Coefficients:")
                print(f"  Natural vs. Manmade: {assort_nat_man:.4f}")
                print(f"  Superclass Groups: {assort_super:.4f}")
                print(f"  Intuitive Groups: {assort_intuitive:.4f}\n")

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

        # Assortativity Coefficients
        # Manmade
        assortativity_values = np.array([[[metrics[epoch][layer][iteration]["assort_nat_man"]
                                       for iteration in iterations] for layer in layers] for epoch in epochs])
        avg_assortativity = np.mean(assortativity_values, axis=2)
        plt.figure(figsize=(10, 6))
        for i, layer in enumerate(layers):
            plt.plot(epochs, avg_assortativity[:, i], label=f'Layer {layer}')
        plt.title('Average Assortativity (Natural vs. Manmade) Across Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Assortativity (Natural vs. Manmade)')
        plt.legend(loc='upper left')
        plt.savefig(f"{output_dir}/assortativity_nat_man.png")

        # Superclass
        assortativity_super_values = np.array([[[metrics[epoch][layer][iteration]["assort_super"]
                                       for iteration in iterations] for layer in layers] for epoch in epochs])
        avg_assortativity_super = np.mean(assortativity_super_values, axis=2)
        plt.figure(figsize=(10, 6))
        for i, layer in enumerate(layers):
            plt.plot(epochs, avg_assortativity_super[:, i], label=f'Layer {layer}')
        plt.title('Average Assortativity (Superclass Groups) Across Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Assortativity (Superclass Groups)')
        plt.legend(loc='upper left')
        plt.savefig(f"{output_dir}/assortativity_super.png")

        # Intuitive
        assortativity_intuitive_values = np.array([[[metrics[epoch][layer][iteration]["assort_intuitive"]
                                       for iteration in iterations] for layer in layers] for epoch in epochs])
        avg_assortativity_intuitive = np.mean(assortativity_intuitive_values, axis=2)
        plt.figure(figsize=(10, 6))
        for i, layer in enumerate(layers):
            plt.plot(epochs, avg_assortativity_intuitive[:, i], label=f'Layer {layer}')
        plt.title('Average Assortativity (Intuitive Groups) Across Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Assortativity (Intuitive Groups)')
        plt.legend(loc='upper left')
        plt.savefig(f"{output_dir}/assortativity_intuitive.png")