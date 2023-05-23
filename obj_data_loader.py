import numpy as np
import dgl.batch as batch
import glob
import trimesh
import networkx as nx
import torch
import dgl


def obj_to_graph(file_path):
    obj = trimesh.load(file_path, process=False, maintain_order=True)
    G = nx.Graph()
    positions = np.array(obj.vertices, dtype=np.float32)
    attributes = [{'position': pos} for pos in positions]
    for idx, attr in enumerate(attributes):
        G.add_node(idx, **attr)
    G.add_edges_from(obj.edges)
    dgl_graph = dgl.from_networkx(G, node_attrs=["position"])
    return dgl_graph


class GraphDataLoader(dgl.data.DGLDataset):
    def __init__(self, obj_directory, data="train"):
        super(GraphDataLoader, self).__init__(name='GraphDataLoader')
        self.obj_directory = obj_directory
        self.graphs = []
        self.labels = []
        self.data = data
        self._load_and_convert()


    def _load_and_convert(self):
        paths = glob.glob("%s/*/%s/*.obj"%(self.obj_directory, self.data))
        for file_name in paths:
            graph, label = self._convert_to_graph(file_name)
            self.graphs.append(graph)
            self.labels.append(label)

    def _convert_to_graph(self, file_path):
        self.label_dict = {
            "annulus": 0,
            "capsule": 1,
            "cone": 2,
            "cube": 3,
            "cylinder": 4,
            "sphere": 5,
        }
        dgl_graph = obj_to_graph(file_path)
        label = self.label_dict[file_path.split("/")[1]]
        return dgl_graph, label

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def collate_fn(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = batch(graphs)
        return batched_graph, torch.tensor(labels)
