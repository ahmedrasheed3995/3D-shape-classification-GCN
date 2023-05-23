import torch
from obj_data_loader import obj_to_graph
from classifier import Classifier


if __name__ == "__main__":

    obj_file_path = "data/cube/train/0.obj"
    model_path = "saved_model.pt"
    model = torch.load(model_path)
    model.eval()

    label_list = [
        "annulus",
        "capsule",
        "cone",
        "cube",
        "cylinder",
        "sphere",
    ]

    dgl_graph = obj_to_graph(obj_file_path)
    feats = dgl_graph.ndata['position']
    logits = model(dgl_graph, feats)
    label = label_list[logits.argmax(dim=1)]

    print("Predicted class: ", label)