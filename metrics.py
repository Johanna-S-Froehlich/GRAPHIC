import argparse
import glob
import importlib
import json
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import file_matrix

try:
    import infomap
except ImportError:  # only needed for the Infomap communities
    infomap = None


# Per-dataset configuration: which mappings module to import, the graph plot
# canvas size that best fits that many class labels, and the groupings the
# assortativity is computed for. Each grouping is (metric suffix, key in the
# module's ``mappings``, name used in prints and plots); the metric is stored
# as ``assort_<suffix>`` and plotted to ``assortativity_<suffix>.png``.
# ``heldout_only_layers`` are the output layers of KimCNN (fc) and Text GCN (2):
# their matrices come from the predictions of the network on the test set, so
# there are none for the training side.
DATASET_CONFIG = {
    "cifar100": {
        "mappings_module": "cifar100_mappings",
        "graph_figsize": (16, 12),
        "assortativity": [
            ("nat_man", "make_superclass", "Natural vs. Manmade"),
            ("super", "superclass_fine", "Superclass Groups"),
            ("intuitive", "intuitive_fine", "Intuitive Groups"),
        ],
    },
    "tiny-imagenet": {
        "mappings_module": "tinyimagenet_mappings",
        "graph_figsize": (32, 24),
        "assortativity": [],
    },
    "r52": {
        "mappings_module": "r52_mappings",
        "graph_figsize": (16, 12),
        "assortativity": [("human", "human", "Human Grouping")],
        "heldout_only_layers": ["fc", "2"],
    },
    "wos46985": {
        "mappings_module": "wos46985_mappings",
        "graph_figsize": (32, 24),
        "assortativity": [("human", "human", "Human Grouping")],
        "heldout_only_layers": ["fc", "2"],
    },
}


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
    Calculate the accuracy score from a row-normalized confusion matrix.
    
    This is the mean per-class recall in percent, which equals the accuracy for
    balanced classes such as CIFAR-100, but not for R52 and WOS-46985.
    
    Parameters:
    C (np.array): Normalized confusion matrix.
    
    Returns:
    float: Accuracy score (between 0 and 100).
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
    a = E.sum(axis=1)  
    b = E.sum(axis=0)   
    expected = a @ b    
    r = (np.trace(E) - expected) / (1 - expected)
    return r

def compute_modularity_louvain(G: nx.DiGraph, seed: int = None) -> float:
    """
    Computes the modularity of a graph using the Louvain modularity optimization method.

    Modularity is a measure of the structure of networks or graphs. It quantifies the strength of division
    of a graph into clusters (communities), where higher values indicate better-defined community structure.

    Args:
        G: nx.DiGraph: A directed graph.
        seed (optional): random seed for Louvain; None gives a random run.

    Returns:
        float: The computed modularity value, a measure of the quality of the graph's community structure.

    """
    # Without any confusion every class is its own community and modularity is undefined.
    if G.number_of_edges() == 0:
        return float('nan'), [{node} for node in G.nodes()]
    # Modularity using Louvain modularity communities
    communities= nx.algorithms.community.louvain_communities(G, weight='weight', seed=seed)
    modularity_louv  = nx.algorithms.community.modularity(G, communities, weight='weight')
    return modularity_louv, communities

def compute_communities_infomap(G: nx.DiGraph, seed: int = None) -> list:
    """
    Detects communities with Infomap on the directed weighted graph.

    Classes without any confusion never enter the Infomap network, so each of
    them is returned as a community of its own.

    Args:
        G: nx.DiGraph: A directed graph with integer nodes 0..n-1.
        seed (optional): random seed for Infomap.

    Returns:
        list[set]: The detected communities.
    """
    if infomap is None:
        raise ImportError("Infomap communities need the infomap package: pip install infomap")
    if G.number_of_edges() == 0:
        return [{node} for node in G.nodes()]

    flags = "--directed --silent" if seed is None else f"--directed --silent --seed {seed}"
    im = infomap.Infomap(flags)
    for u, v, data in G.edges(data=True):
        im.add_link(u, v, data['weight'])
    im.run()

    modules = {}
    for node in im.tree:
        if node.is_leaf:
            modules.setdefault(node.module_id, set()).add(node.node_id)
    communities = list(modules.values())
    assigned = set().union(*communities)
    communities += [{node} for node in G.nodes() if node not in assigned]
    return communities

def compute_communities_label_propagation(C_directed: np.ndarray, seed: int = None) -> list:
    """
    Detects communities with weighted asynchronous label propagation.

    Label propagation has no directed formulation, so it runs on the symmetrized
    graph 0.5 * (C + C^T), which keeps the confusions of both directions.

    Args:
        C_directed: Confusion matrix without its diagonal.
        seed (optional): random seed for label propagation.

    Returns:
        list[set]: The detected communities.
    """
    G_sym = nx.from_numpy_array(0.5 * (C_directed + C_directed.T))
    return list(nx.algorithms.community.asyn_lpa_communities(G_sym, weight='weight', seed=seed))

def compute_ami(communities: list, fine_labels: list[str], human_grouping: list[set]) -> float:
    """
    Computes the adjusted mutual information between detected communities and a human grouping.

    Args:
        communities: detected communities as sets of class indices.
        fine_labels: class names, in confusion-matrix order.
        human_grouping: the human grouping as sets of class names.

    Returns:
        float: Adjusted mutual information (1 for identical groupings, about 0 for chance).
    """
    # Imported here so that the image pipeline does not load scikit-learn.
    from sklearn.metrics import adjusted_mutual_info_score

    human_id = {label: g for g, group in enumerate(human_grouping) for label in group}
    detected_id = {node: c for c, community in enumerate(communities) for node in community}
    # Compare only classes that are in the human grouping and in the graph.
    compared = [i for i, label in enumerate(fine_labels) if label in human_id and i in detected_id]
    human = [human_id[fine_labels[i]] for i in compared]
    detected = [detected_id[i] for i in compared]
    return adjusted_mutual_info_score(human, detected)

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

def plot_graph(G: nx.DiGraph, filepath: str, communities: list[list[int]]=None,
               colors_communities: list[list[float]]=None, title: str=None,
               node_names: list[str]=None, figsize: tuple[int, int]=(16, 12)):
    """Plot a picture of a graph and store it a defined place.
    Args:
        G: Graph to be plotted.
        filepath: path to file where the result should be written
        communities (optional): indicated the identified communities by color. Leave None for detection of communities by modularity.
        colors_communities (optional): list of colors for each community, if None, random colors will be used.
        title (optional): title of the graph
        figsize (optional): matplotlib figure size; larger values fit more
            labels (use (32, 24) for ~200-class datasets like Tiny ImageNet).
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
    plt.figure(figsize=figsize)
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


def _plot_metric_over_epochs(metric_key: str, title: str, ylabel: str, output_path: str,
                             metrics_dict, epochs, layers, iterations):
    """Helper: plot one scalar metric averaged over iterations, one line per layer."""
    values = np.array([[[metrics_dict[epoch][layer][iteration][metric_key]
                         for iteration in iterations] for layer in layers] for epoch in epochs])
    avg = np.nanmean(values, axis=2)
    plt.figure(figsize=(10, 6))
    for i, layer in enumerate(layers):
        plt.plot(epochs, avg[:, i], label=f'Layer {layer}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.savefig(output_path)


# Main Script
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate target metrics for confusion matrices.')
    parser.add_argument('--matrix-dir', type=str, required=True,
                        help='Directory containing the confusion matrix files.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the output metrics/plots.')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=sorted(DATASET_CONFIG.keys()),
                        help='Dataset whose class metadata to use. Tiny ImageNet '
                             'has no class hierarchy, so assortativity is skipped. '
                             'R52 and WOS-46985 also get the AMI with the human grouping.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The parameter determining the cross entropy mix between true and predicted labels.')
    parser.add_argument('--train-set', action='store_true', help='Use training set confusion matrices.')
    parser.add_argument('--plot-metrics', action='store_true', help='Plot metrics after computation.')
    parser.add_argument('--plot-graphs', action='store_true', help='Plot graphs according to modularity communities.')
    parser.add_argument('--all-communities', action='store_true',
                        help='Also detect communities with Infomap and label propagation '
                             '(default: Louvain only).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for Louvain, Infomap and label propagation.')
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]
    mappings_mod = importlib.import_module(config["mappings_module"])
    fine_labels = mappings_mod.fine_labels
    # Only datasets with a hierarchy expose ``mappings`` (one dict per grouping).
    dataset_mappings = getattr(mappings_mod, "mappings", None)
    assortativity_groupings = config["assortativity"] if dataset_mappings is not None else []
    # Datasets with a human baseline grouping expose ``human_grouping``.
    human_grouping = getattr(mappings_mod, "human_grouping", None)
    algorithms = ["louvain", "infomap", "labelprop"] if args.all_communities else ["louvain"]
    algorithm_names = {"louvain": "Louvain", "infomap": "Infomap", "labelprop": "Label Propagation"}
    num_classes = len(fine_labels)

    output_dir = os.path.join(args.output_dir, f"alpha_{args.alpha}")
    os.makedirs(output_dir, exist_ok=True)

    matrix_dir = os.path.join(args.matrix_dir, f"alpha_{args.alpha}")
    matrix_files = glob.glob(f"{matrix_dir}/*.npy")
    if not matrix_files:
        raise FileNotFoundError(f"No confusion matrix files found in {matrix_dir}")

    metrics = {}

    fm, epochs, iterations, layers = file_matrix(matrix_files)
    print(f"Found {len(epochs)} epochs, {len(iterations)} iterations, and {len(layers)} layers.")

    split_index = 0 if args.train_set else 1
    # The output layers of the text models are skipped for the training side.
    # Any other gap means the directory is incomplete.
    available = fm[:, :, :, split_index] != ""
    skipped = config.get("heldout_only_layers", []) if args.train_set else []
    layer_indices = [j for j, layer in enumerate(layers)
                     if layer not in skipped or available[:, :, j].any()]
    if not layer_indices:
        raise FileNotFoundError(f"No confusion matrices for this partition in {matrix_dir}")
    missing = [f"epoch {epochs[i]}, iteration {iterations[k]}, layer {layers[j]}"
               for i, k, j in zip(*np.nonzero(~available)) if j in layer_indices]
    if missing:
        raise FileNotFoundError(f"Missing confusion matrices in {matrix_dir}: " + "; ".join(missing))
    fm = fm[:, :, layer_indices, :]
    layers = [layers[j] for j in layer_indices]

    if args.plot_graphs:
        graph_dir = os.path.join(args.output_dir, f"alpha_{args.alpha}", "graphs")
        colors = community_colors(num_classes)

    # Iterate through epochs, iterations, and layers
    for (i, epoch) in enumerate(epochs):
        metrics[epoch] = {}
        for (j, layer) in enumerate(layers):
            metrics[epoch][layer] = {}
            for (k, iteration) in enumerate(iterations):
                # Get the confusion matrix file for the current epoch, layer, and iteration
                matrix_file = fm[i, k, j, split_index]

                C = np.load(matrix_file)
                if C.ndim != 2 or C.shape[0] != C.shape[1]:
                    raise ValueError(f"Invalid confusion matrix shape in file {matrix_file}: {C.shape}")

                C_directed = C - np.diag(np.diag(C))

                # Create graph from the matrix.
                G_directed = nx.from_numpy_array(C_directed, create_using=nx.DiGraph())

                modularity_louvain, communities_louvain = compute_modularity_louvain(G_directed, seed=args.seed)
                communities = {"louvain": communities_louvain}
                if args.all_communities:
                    communities["infomap"] = compute_communities_infomap(G_directed, seed=args.seed)
                    communities["labelprop"] = compute_communities_label_propagation(C_directed, seed=args.seed)
                accuracy = calculate_accuracy_score(C)

                entry = {"modularity_louvain": float(modularity_louvain)}
                for algorithm in algorithms:
                    entry[f"nr_communities_{algorithm}"] = int(len(communities[algorithm]))
                entry["accuracy"] = float(accuracy)

                if human_grouping is not None:
                    for algorithm in algorithms:
                        entry[f"ami_{algorithm}_human"] = float(
                            compute_ami(communities[algorithm], fine_labels, human_grouping))

                for suffix, grouping_key, _ in assortativity_groupings:
                    entry[f"assort_{suffix}"] = float(compute_weighted_assortativity(
                        C_directed, fine_labels, dataset_mappings[grouping_key]))

                metrics[epoch][layer][iteration] = entry

                if args.plot_graphs:
                    file_path = os.path.join(graph_dir, f'epoch_{epochs[i]}_iteration_{iterations[k]}_layer_{layers[j]}.svg')
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    plot_graph(G_directed, filepath=file_path, communities=communities_louvain,
                               colors_communities=colors,
                               title=f'Epoch {epochs[i]} Iteration {iterations[k]} Layer {layers[j]}',
                               node_names=fine_labels, figsize=config["graph_figsize"])

                print("=== Graph Analysis Metrics ===")
                print(f"Epoch: {epoch}, Layer: {layer}, Iteration: {iteration}")
                print(f"Modularity (Louvain): {modularity_louvain:.4f}")
                for algorithm in algorithms:
                    print(f"Number of {algorithm_names[algorithm]} Communities: "
                          f"{entry[f'nr_communities_{algorithm}']}")
                print()
                if human_grouping is not None:
                    print("AMI with the human grouping:")
                    for algorithm in algorithms:
                        print(f"  {algorithm_names[algorithm]}: {entry[f'ami_{algorithm}_human']:.4f}")
                    print()
                if assortativity_groupings:
                    print("Assortativity Coefficients:")
                    for suffix, _, name in assortativity_groupings:
                        print(f"  {name}: {entry[f'assort_{suffix}']:.4f}")
                    print()
                print(f"Accuracy: {accuracy:.4f}")


    # Save metrics to json file
    output_file = f"{output_dir}/graph_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if args.plot_metrics:
        _plot_metric_over_epochs(
            "modularity_louvain", "Average Modularity (Louvain) Across Epochs",
            "Modularity (Louvain)", f"{output_dir}/modularity_louvain.png",
            metrics, epochs, layers, iterations,
        )
        for algorithm in algorithms:
            name = algorithm_names[algorithm]
            if args.all_communities or human_grouping is not None:
                _plot_metric_over_epochs(
                    f"nr_communities_{algorithm}",
                    f"Average Number of Communities ({name}) Across Epochs",
                    f"Number of Communities ({name})",
                    f"{output_dir}/nr_communities_{algorithm}.png",
                    metrics, epochs, layers, iterations,
                )
            if human_grouping is not None:
                _plot_metric_over_epochs(
                    f"ami_{algorithm}_human",
                    f"Average AMI with the Human Grouping ({name}) Across Epochs",
                    f"AMI ({name} vs. human)",
                    f"{output_dir}/ami_{algorithm}_human.png",
                    metrics, epochs, layers, iterations,
                )
        for suffix, _, name in assortativity_groupings:
            _plot_metric_over_epochs(
                f"assort_{suffix}",
                f"Average Assortativity ({name}) Across Epochs",
                f"Assortativity ({name})",
                f"{output_dir}/assortativity_{suffix}.png",
                metrics, epochs, layers, iterations,
            )
