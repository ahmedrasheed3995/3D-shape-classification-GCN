import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
import matplotlib.pyplot as plt
from obj_data_loader import GraphDataLoader


def data_loaders(obj_directory, batch_size=6):
    # Train
    train_loader = GraphDataLoader(obj_directory, data="train")
    data_loader_train = dgl.dataloading.GraphDataLoader(
        train_loader,
        batch_size=batch_size,
        collate_fn=train_loader.collate_fn
    )

    # Validation
    val_loader = GraphDataLoader(obj_directory, data="validation")
    data_loader_val = dgl.dataloading.GraphDataLoader(
        val_loader,
        batch_size=batch_size,
        collate_fn=val_loader.collate_fn
    )

    # Test
    test_loader = GraphDataLoader(obj_directory, data="test")
    data_loader_test = dgl.dataloading.GraphDataLoader(
        test_loader,
        batch_size=batch_size,
        collate_fn=test_loader.collate_fn
    )
    return data_loader_train, data_loader_val, data_loader_test

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


def train(model, train, validation, optimizer, epochs, save_path="saved_model.pt", plot=True):
    val_loss_list = []
    val_acc_list = []
    train_loss_list = []
    train_acc_list = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        iterations = 0
        iterations_val = 0
        for batched_graph, labels in train:
            feats = batched_graph.ndata['position']
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_accuracy += acc(logits.argmax(dim=1), labels).item()
            iterations += 1

        for batched_graph, labels in validation:
            with torch.no_grad():
                feats = batched_graph.ndata['position']
                logits = model(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                epoch_val_loss += loss.detach().item()
                epoch_val_accuracy += acc(logits.argmax(dim=1), labels).item()
                iterations_val += 1

        if plot:
            train_loss_list.append(epoch_loss/iterations)
            train_acc_list.append(epoch_accuracy/iterations)

            val_loss_list.append(epoch_val_loss/iterations_val)
            val_acc_list.append(epoch_val_accuracy/iterations_val)

        print("Epoch %3i -- Loss >> Train %.4f - Val %.4f -- Acc >> Train %.4f - Val %.4f" % (
            epoch,
            epoch_loss/iterations,
            epoch_val_loss/iterations_val,
            epoch_accuracy / iterations,
            epoch_val_accuracy / iterations_val,
        ))

    # model_scripted = torch.jit.script(model)
    # model_scripted.save('model_scripted.pt')
    torch.save(model, save_path)

    if plot:
        # Plot loss
        plt.figure()
        plt.plot(train_loss_list, label='Training Loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.show()

        # Plot accuracy
        plt.figure()
        plt.plot(train_acc_list, label='Training Accuracy')
        plt.plot(val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.show()

    return model


def test(model, test):
    test_accuracy = 0
    iterations_test = 0
    for batched_graph, labels in test:
        with torch.no_grad():
            feats = batched_graph.ndata['position']
            logits = model(batched_graph, feats)
            test_accuracy += acc(logits.argmax(dim=1), labels).item()
            iterations_test += 1

    return test_accuracy/iterations_test


if __name__ == "__main__":
    obj_directory = 'data'
    batch_size = 6
    train_data, val_data, test_data = data_loaders(obj_directory, batch_size)
    model = Classifier(in_dim=3, hidden_dim=16, n_classes=6)
    opt = torch.optim.Adam(model.parameters())
    acc = Accuracy(task="multiclass", num_classes=6)
    model = train(
        model=model,
        train=train_data,
        validation=val_data,
        optimizer=opt,
        epochs=150,
        plot=True,
    )
    test_acc = test(model, test_data)
    print("test_acc: ", test_acc)
