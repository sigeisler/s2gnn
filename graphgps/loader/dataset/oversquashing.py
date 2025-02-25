from itertools import product
import os.path as osp
import random
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer


# Start from https://github.com/lrnzgiusti/on-oversquashing/blob/main/data/ring_transfer.py
def generate_ring_transfer_graph(nodes, target_label,
                                 add_crosses: bool = False):
    """
    Generate a ring transfer graph with an option to add crosses.

    Args:
    - nodes (int): Number of nodes in the graph.
    - target_label (list): Label of the target node.
    - add_crosses (bool): Whether to add cross edges in the ring.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    assert nodes > 1, ValueError("Minimum of two nodes required")
    # Determine the node directly opposite to the source (node 0) in the ring
    opposite_node = nodes // 2

    # Initialise feature matrix with a uniform feature.
    # This serves as a placeholder for features of all nodes.
    x = np.ones((nodes, len(target_label)))

    # Set feature of the source node to 0 and the opposite node to the target label
    x[0, :] = 0.0
    x[opposite_node, :] = target_label

    # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
    x = torch.tensor(x, dtype=torch.float32)

    # List to store edge connections in the graph
    edge_index = []
    for i in range(nodes - 1):
        # Regular connections that make the ring
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

        # Conditionally add cross edges, if desired
        if add_crosses and i < opposite_node:
            # Add edges from a node to its direct opposite
            edge_index.append([i, nodes - 1 - i])
            edge_index.append([nodes - 1 - i, i])

            # Extra logic for ensuring additional "cross" edges in some conditions
            if nodes + 1 - i < nodes:
                edge_index.append([i, nodes + 1 - i])
                edge_index.append([nodes + 1 - i, i])

    # Close the ring by connecting the last and the first nodes
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    # Convert edge list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to identify the target node in the graph. Only the source node (index 0) is marked true.
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1

    # Determine the graph's label based on the target label. This is a singular value indicating the index of the target label.
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

    # Return the graph with nodes, edges, mask and the label
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_lollipop_transfer_graph(nodes: int, target_label: List[int]):
    """
    Generate a lollipop transfer graph.

    Args:
    - nodes (int): Total number of nodes in the graph.
    - target_label (list): Label of the target node.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    # Initialize node features. The first node gets 0s, while the last gets the target label
    x = np.ones((nodes, len(target_label)))
    x[0, :] = 0.0
    x[nodes - 1, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    n_d_2_trunc = min(nodes // 2, 15)

    # Construct a clique for the first half of the nodes,
    # where each node is connected to every other node except itself
    for i in range(n_d_2_trunc):
        for j in range(n_d_2_trunc):
            if i == j:  # Skip self-loops
                continue
            edge_index.append([i, j])
            edge_index.append([j, i])

    # Construct a path (a sequence of connected nodes) for the second half of the nodes
    for i in range(n_d_2_trunc, nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    # Connect the last node of the clique to the first node of the path
    edge_index.append([n_d_2_trunc - 1, n_d_2_trunc])
    edge_index.append([n_d_2_trunc, n_d_2_trunc - 1])

    # Convert the edge index list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to indicate the target node (in this case, the first node)
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1

    # Convert the one-hot encoded target label to its corresponding class index
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, mask=mask, y=y)
# End


class OverSquashingDataset(InMemoryDataset):
    def __init__(self,
                 n_graphs=(5_000, 500, 5_000),
                 gen_mode='full',  # or random
                 train_n_nodes=(4, 50),
                 valid_n_nodes=(4, 50),
                 test_n_nodes=(52, 100),
                 n_classes: int = 5,
                 topology='ring_lollipop',  # or ring, lollipop
                 seed=242,
                 root='datasets'):
        self.original_root = root
        self.gen_mode = gen_mode
        dataset_name = (
            f'over_squashing_{topology}_{n_classes}classes_{self.gen_mode}_' +
            f'{train_n_nodes[0]}to{train_n_nodes[1] - 1}_' +
            f'{valid_n_nodes[0]}to{valid_n_nodes[1] - 1}_' +
            f'{test_n_nodes[0]}to{test_n_nodes[1] - 1}')
        self.folder = osp.join(root, 'over-squashing', dataset_name)
        self.train_dist = (train_n_nodes[0] // 2, train_n_nodes[1] // 2 + 1)
        self.valid_dist = (valid_n_nodes[0] // 2, valid_n_nodes[1] // 2 + 1)
        self.test_dist = (test_n_nodes[0] // 2, test_n_nodes[1] // 2 + 1)
        mul = len(topology.split('_')) * n_classes
        if self.gen_mode == 'full':
            self.n_graphs = dict(
                train=(self.train_dist[1] - self.train_dist[0]) * mul,
                valid=(self.valid_dist[1] - self.valid_dist[0]) * mul,
                test=(self.test_dist[1] - self.test_dist[0]) * mul)
        else:
            self.n_graphs = dict(zip(['train', 'valid', 'test'], n_graphs))
        self.train_n_nodes = train_n_nodes
        self.valid_n_nodes = valid_n_nodes
        self.test_n_nodes = test_n_nodes
        self.n_classes = n_classes
        self.topology = topology
        self.random = np.random.RandomState(seed)

        super().__init__(self.folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        data_list = []
        for split, n_graphs in self.n_graphs.items():
            print(f'Generating {n_graphs} graphs for {split} split...')
            if self.gen_mode == 'random':
                for _ in tqdm(range(n_graphs)):
                    dist = self.random.choice(
                        range(*getattr(self, f'{split}_dist')))
                    label = self.random.randint(0, self.n_classes)

                    topology = self.topology
                    if '_' in topology:
                        topology = self.random.choice(topology.split('_'))
                    data = self.generate_graph(split, dist, label, topology)
                    data_list.append(data)
            else:
                vals = product(list(range(*getattr(self, f'{split}_dist'))),
                               self.topology.split('_'),
                               list(range(self.n_classes)))
                vals = list(vals)
                self.random.shuffle(vals)
                for dist, topology, label in tqdm(vals):
                    data = self.generate_graph(split, dist, label, topology)
                    data_list.append(data)

        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        n_train, n_valid, n_test = self.n_graphs.values()
        return {
            'train': torch.arange(n_train),
            'valid': torch.arange(n_valid) + n_train,
            'test': torch.arange(n_test) + n_train + n_valid,
        }

    def generate_graph(self, split, dist, label, topology):
        n_nodes = 2 * dist

        target_class = np.zeros(self.n_classes)
        target_class[label] = 1.0

        topology = self.topology
        if '_' in topology:
            topology = self.random.choice(topology.split('_'))

        if topology == 'ring':
            graph = generate_ring_transfer_graph(n_nodes, target_class)
        elif topology == 'lollipop':
            graph = generate_lollipop_transfer_graph(n_nodes, target_class)
        else:
            raise ValueError(f'Topology {topology} not supported')
        return graph


if __name__ == '__main__':
    dataset = OverSquashingDataset()


# dataset_factory = {
#         'TREE': generate_tree_transfer_graph_dataset,
#         'RING': generate_ring_transfer_graph_dataset,
#         'LOLLIPOP': generate_lollipop_transfer_graph_dataset
#     }

# dataset_configs = {
#         'depth': args.synthetic_size,
#         'nodes': args.synthetic_size,
#         'classes': args.num_class,
#         'samples': args.synth_train_size + args.synth_test_size,
#         'arity': args.arity,
#         'add_crosses': int(args.add_crosses)
#     }

# parser.add_argument('--add_crosses', type=str2bool, default=False)
# parser.add_argument('--synth_train_size', type=int, default=5000)
# parser.add_argument('--synth_test_size', type=int, default=500)
# parser.add_argument('--synthetic_size', type=int, default=10)
# parser.add_argument('--generate_tree', type=str2bool, default=False)
# parser.add_argument('--arity', type=int, default=2)
# parser.add_argument('--num_class', type=int, default=5)


def generate_ring_lookup_graph(nodes: int):
    """
    Generate a dictionary lookup ring graph.

    Args:
    - nodes (int): Number of nodes in the ring.

    Returns:
    - Data: Torch geometric data structure containing graph details.

    Note: This function is currently deprecated.
    """

    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    # Generate unique keys and random values for all the nodes except the source node
    # Create an array of unique keys from 1 to nodes-1
    keys = np.arange(1, nodes)
    # Shuffle these keys to serve as values
    vals = np.random.permutation(nodes - 1)

    # One-hot encode keys and values
    oh_keys = np.array(LabelBinarizer().fit_transform(keys))
    oh_vals = np.array(LabelBinarizer().fit_transform(vals))

    # Concatenate one-hot encoded keys and values to create node features
    oh_all = np.concatenate((oh_keys, oh_vals), axis=-1)
    x = np.empty((nodes, oh_all.shape[1]))
    x[1:, :] = oh_all  # Add these as features for all nodes except the source node

    # Randomly select one key for the source node and associate a random value to it
    key_idx = random.randint(0, nodes - 2)  # Random index for choosing a key
    val = vals[key_idx]  # Corresponding value from the randomized list

    # Set the source node's features: all zeros except the chosen key which is set to one-hot encoded value
    x[0, :] = 0
    # Assigning one-hot encoded key to source node
    x[0, :oh_keys.shape[1]] = oh_keys[key_idx]

    # Convert to tensor for PyTorch compatibility
    x = torch.tensor(x, dtype=torch.float32)

    # Generate edges for the ring topology
    edge_index = []
    for i in range(nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    # Add the edges to complete the ring
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    # Convert to tensor and transpose for Torch Geometric compatibility
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to highlight the source node (used later for graph-level predictions)
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1  # Source node is the node of interest

    # Add a label based on the random value associated with the source node's key
    y = torch.tensor([val], dtype=torch.long)

    # Return a Torch Geometric Data object containing all graph information
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ringlookup_graph_dataset(nodes: int, samples: int = 10000):
    """
    Generate a dataset of ring lookup graphs.

    Args:
    - nodes (int): Number of nodes in each graph.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    dataset = []
    for i in range(samples):
        graph = generate_ring_lookup_graph(nodes)
        dataset.append(graph)
    return dataset


def generate_ring_transfer_graph(nodes, target_label,
                                 add_crosses: bool = False):
    """
    Generate a ring transfer graph with an option to add crosses.

    Args:
    - nodes (int): Number of nodes in the graph.
    - target_label (list): Label of the target node.
    - add_crosses (bool): Whether to add cross edges in the ring.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    assert nodes > 1, ValueError("Minimum of two nodes required")
    # Determine the node directly opposite to the source (node 0) in the ring
    opposite_node = nodes // 2

    # Initialise feature matrix with a uniform feature.
    # This serves as a placeholder for features of all nodes.
    x = np.ones((nodes, len(target_label)))

    # Set feature of the source node to 0 and the opposite node to the target label
    x[0, :] = 0.0
    x[opposite_node, :] = target_label

    # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
    x = torch.tensor(x, dtype=torch.float32)

    # List to store edge connections in the graph
    edge_index = []
    for i in range(nodes - 1):
        # Regular connections that make the ring
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

        # Conditionally add cross edges, if desired
        if add_crosses and i < opposite_node:
            # Add edges from a node to its direct opposite
            edge_index.append([i, nodes - 1 - i])
            edge_index.append([nodes - 1 - i, i])

            # Extra logic for ensuring additional "cross" edges in some conditions
            if nodes + 1 - i < nodes:
                edge_index.append([i, nodes + 1 - i])
                edge_index.append([nodes + 1 - i, i])

    # Close the ring by connecting the last and the first nodes
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    # Convert edge list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask to identify the target node in the graph. Only the source node (index 0) is marked true.
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[0] = 1

    # Determine the graph's label based on the target label. This is a singular value indicating the index of the target label.
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

    # Return the graph with nodes, edges, mask and the label
    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_ring_transfer_graph_dataset(nodes: int, add_crosses: bool = False,
                                         classes: int = 5,
                                         samples: int = 10000, **kwargs):
    """
    Generate a dataset of ring transfer graphs.

    Args:
    - nodes (int): Number of nodes in each graph.
    - add_crosses (bool): Whether to add cross edges in the ring.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_ring_transfer_graph(nodes, target_class, add_crosses)
        dataset.append(graph)
    return dataset


def generate_tree_transfer_graph(depth: int, target_label: List[int],
                                 arity: int):
    """
    Generate a tree transfer graph.

    Args:
    - depth (int): Depth of the tree.
    - target_label (list): Label of the target node.
    - arity (int): Number of children each node can have.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    if depth <= 0:
        raise ValueError("Minimum of depth one")
    # Calculate the total number of nodes based on the tree depth and arity
    num_nodes = int((arity ** (depth + 1) - 1) / (arity - 1))

    # Target node is the last node in the tree
    target_node = num_nodes - 1

    # Initialize the feature matrix with a constant feature vector
    x = np.ones((num_nodes, len(target_label)))

    # Set root node and target node feature values
    x[0, :] = 0.0
    x[target_node, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    # To keep track of the current child node while iterating
    last_child_counter = 0

    # Loop to generate the edges of the tree
    for i in range(num_nodes - arity ** depth + 1):
        for child in range(1, arity + 1):
            # Ensure we don't exceed the total number of nodes
            if last_child_counter + child > num_nodes - 1:
                break

            # Add edges for the current node and its children
            edge_index.append([i, last_child_counter + child])
            edge_index.append([last_child_counter + child, i])

        # Update the counter to point to the last child of the current node
        last_child_counter += arity

    # Convert edge index to torch tensor format
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask for the root node of the graph
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[0] = 1

    # Convert the target label into a single value format
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, mask=mask, y=y)


def generate_tree_transfer_graph_dataset(depth: int, arity: int,
                                         classes: int = 5,
                                         samples: int = 10000, **kwargs):
    """
    Generate a dataset of tree transfer graphs.

    Args:
    - depth (int): Depth of the tree in each graph.
    - arity (int): Number of children each node can have.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_tree_transfer_graph(depth, target_class, arity)
        dataset.append(graph)
    return dataset


def generate_lollipop_transfer_graph_dataset(nodes: int, classes: int = 5,
                                             samples: int = 10000, **kwargs):
    """
    Generate a dataset of lollipop transfer graphs.

    Args:
    - nodes (int): Total number of nodes in each graph.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_lollipop_transfer_graph(nodes, target_class)
        dataset.append(graph)
    return dataset
