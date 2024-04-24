from pathlib import Path
from typing import Iterable
import numpy as np
from torch_geometric.loader import DataLoader
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix,
    MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassROC,
    MulticlassPrecisionRecallCurve
    )
from datetime import datetime
from dateutil import tz
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch import no_grad, tensor
from torchmetrics.wrappers import MetricTracker
import matplotlib.pyplot as plt
from torch_geometric.transforms import BaseTransform, RemoveIsolatedNodes
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkit as nk
import pandas as pd
from utils import get_now_datetime
import networkx as nx


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta_factor: float = 0.01):
        self.patience = patience
        self.min_delta_factor = min_delta_factor
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float):
        min_delta = self.min_delta_factor * validation_loss
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomRandomNodeSplitMasker(BaseTransform):
    def __init__(self, test_size, validation_size, random_state, log=False):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.log = log

    def forward(self, data):
        if data.is_cuda:
            device = torch.device('cpu')
            data = data.to(device)
            num_nodes = data.num_nodes
            labels = data.y.numpy()
            device = torch.device('cuda')
            data = data.to(device)
        else:
            num_nodes = data.num_nodes
            labels = data.y.numpy()


        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)

        # Split the data indices into train and test sets
        for train_index, test_index in splitter.split(np.zeros(num_nodes), labels):
            train_indices, test_indices = train_index, test_index

        # Now, split the training indices into training and validation sets
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_size, random_state=self.random_state)

        for train_index, val_index in splitter.split(labels[train_indices], labels[train_indices]):
            train_indices, val_indices = train_index, val_index

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        # set indices where to true for nodes to include from indices in train_indices
        train_mask[train_indices] = True

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_indices] = True

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = True

        if self.log:
            print(f"Training set: 0")
            print(f"Validation set: 1")
            print(f"Test set: 2")
            masks = [train_mask, val_mask, test_mask]

            for idx, mask in enumerate(masks):
                print(f"Number of nodes in set {idx}: {mask.sum()}")
                element_counts = Counter(labels[mask])
                element_counts_dict = dict(element_counts)
                print(f"Classes in set {idx}:", element_counts_dict)
        
        # Assign masks to data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data
    
class CustomRandomNodeUnderSampler(BaseTransform):
    def __init__(self, random_state, allowSelfLoops: bool = False, removeIsolatedNodes: bool = True, isolatedRemoverStrategy: BaseTransform = RemoveIsolatedNodes()):
        self.random_state = random_state
        self.allowSelfLoops = allowSelfLoops
        self.removeIsolatedNodes = removeIsolatedNodes
        self.isolatedRemoverStrategy = isolatedRemoverStrategy
    def forward(self, data):
        wasCuda = data.is_cuda
        if wasCuda:
            device = torch.device('cpu')
            data = data.to(device)
            labels = data.y.numpy()
        else:
            labels = data.y.numpy()
        
        underSampler = RandomUnderSampler(random_state=self.random_state)
        dummy_x = np.zeros_like(labels)
        dummy_x_reshaped = dummy_x.reshape(-1, 1)
        _, __ = underSampler.fit_resample(dummy_x_reshaped, labels)
        indices_keep = underSampler.sample_indices_

        nodes_to_keep = [data.x[node_idx] for node_idx in indices_keep]
        nodes_to_keep_labels = [labels[node_idx] for node_idx in indices_keep]
        indices_keep_dict = {node:idx for idx, node in enumerate(indices_keep)}

        # Graph connectivity in COO format. shape [2, num_edges]
        #  "top" list is source node, "bottom" list is destination node
        # edges are based on index position in all_words_ordered
        edges_keep: list[list] = [[], []]
        for edge_pair_idx in range(len(data.edge_index[0])):
            source_node = int(data.edge_index[0][edge_pair_idx])
            destination_node = int(data.edge_index[1][edge_pair_idx])
            if (source_node == destination_node) and not self.allowSelfLoops:
                continue
            if source_node in indices_keep_dict and destination_node in indices_keep_dict:
                source_node_new_idx = indices_keep_dict[source_node]
                destination_node_new_idx = indices_keep_dict[destination_node]
                edges_keep[0].append(source_node_new_idx)
                edges_keep[1].append(destination_node_new_idx)
        
        # pd.DataFrame(edges_keep).to_csv("edges_keep.csv")

        remaining_data = Data(
            x=torch.tensor(np.asarray(nodes_to_keep), dtype=torch.float32),
            edge_index=torch.tensor(edges_keep),
            y=torch.tensor(np.asarray(nodes_to_keep_labels), dtype=torch.int64)
        )

        if self.removeIsolatedNodes:
            remaining_data = self.isolatedRemoverStrategy(remaining_data)

        if wasCuda:
            device = torch.device('cuda')
            remaining_data = remaining_data.to(device)
        
        return remaining_data

class ConcatNodeCentralities(BaseTransform):
    def __init__(self) -> None:
        return
    
    def forward(self, data):
        wasCuda = data.is_cuda
        if data.is_cuda:
            device = torch.device('cpu')
            data = data.to(device)
        nk.setNumberOfThreads(12) 
        # ERROR
        # error in conversion to networkx adds mode nodes that in original 1665 nodes to be exact
        graph_nx = to_networkx(data)
        G_nk = nk.nxadapter.nx2nk(graph_nx)

        degree_centrality = nk.centrality.DegreeCentrality(G_nk, normalized=True).run().scores()
        closeness_centrality = nk.centrality.Closeness(G_nk, True, True).run().scores()
        betweenness_centrality = nk.centrality.Betweenness(G_nk).run().scores()
        eigen_centrality = nk.centrality.EigenvectorCentrality(G_nk).run().scores()

        centralities = {
            "degree_centralities": degree_centrality,
            "closeness_centralities": closeness_centrality,
            "betweenness_centralities": betweenness_centrality,
            "eigen_centralities": eigen_centrality
        }

        centralities_df = pd.DataFrame(centralities)
        centralities_array = centralities_df.to_numpy(dtype=np.float32)
        centralities_tensor = tensor(centralities_array, dtype=torch.float32)
        concat_features = torch.cat((data.x, centralities_tensor), dim=1)
        concat_features_df = pd.DataFrame(concat_features)
        # concat_features_df.to_csv("concat_features_df.csv")

        transformed_node_feature_graph = Data(
            x = concat_features,
            edge_index = data.edge_index,
            y = data.y,
        )

        if wasCuda:
            device = torch.device('cuda')
            transformed_node_feature_graph = transformed_node_feature_graph.to(device)
        
        return transformed_node_feature_graph


def generateAllPlots(metrics: Iterable[Metric], savePathDirectory: str | None = None):
    for metric in metrics:
        if "score" in metric.plot.__annotations__:
            fig, ax = metric.plot(score=True)
        else:
            fig, ax = metric.plot()
            lines = ax.lines
            # very hard coded
            if len(lines) == 1 or len(line.get_ydata() == 7):
                for line in lines:
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    
                    # Annotate each point with its value
                    for xi, yi in zip(x_data, y_data):
                        ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

        if savePathDirectory:
            fig.savefig(f"{savePathDirectory}/{metric.__class__.__name__}.png")
        plt.show()

def plotTrainVSValidationLoss(trainLoss: list[float], valLoss: list[float], savePathDirectory: str | None =None):
    fig, ax = plt.subplots()
    ax.plot(tensor(trainLoss), label='Training loss')
    ax.plot(tensor(valLoss), label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss percent')
    ax.set_title('Train Loss V.S. Validation Loss')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()  # Show legend if labels are provided
    # Add a horizontal line for every y-tick
    for y_tick in ax.get_yticks():
        ax.axhline(y_tick, color='gray', linestyle='-', linewidth=0.5)

    # Add a dashed horizontal line at y=0.0
    ax.axhline(0.0, color='grey', linestyle='--', linewidth=2)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=2)
    ax.text(0.2, 0.002, 'Optimal\nvalue', color='black', fontsize=10, ha='center', va='center')
    if savePathDirectory:
        fig.savefig(f"{savePathDirectory}/{fig.axes[0].get_title()}.png")
    plt.show()

def saveExperiment(output_dir: str, averageTrainLoesses: list, averageValLosses: list, metricCollection: MetricCollection):
    pst = tz.gettz('America/Los_Angeles')

    now_datetime = datetime.now(pst)
    # D=date, T=time
    date_time_str = now_datetime.strftime("D%m-%dT%H_%M_%S")

    EXPERIMENT_DIRECTORY_NAME = f"{output_dir}/GCN_lol{date_time_str}"

    Path(EXPERIMENT_DIRECTORY_NAME).mkdir(parents=True, exist_ok=False)
    # loss graph
    plotTrainVSValidationLoss(averageTrainLoesses, averageValLosses, EXPERIMENT_DIRECTORY_NAME)
    # Extract all metrics from all steps
    generateAllPlots(metricCollection.values(), EXPERIMENT_DIRECTORY_NAME)


def testPhase(model: Module, criterion: CrossEntropyLoss, test_loader: DataLoader, device):
    model.eval()
    test_accuracies = []
    test_losses = []
    with no_grad():
        for batch in test_loader:
            test_loss = 0.0
            test_correct = 0.0
            # Similar to the validation loop, perform forward pass, compute loss, and accuracy
            # batch = batch.to(device)

            # outputs = model(batch.x, batch.edge_index)
            # loss = criterion(outputs[batch.test_mask], batch.y[batch.test_mask])

            # pred = outputs.argmax(dim=1)  # Use the class with the highest probability.

            output, h = model(batch.x, batch.edge_index)
            label = batch.y[batch.test_mask]
            pred = output[batch.test_mask]
            loss = criterion(pred, label)
            pred = pred.argmax(dim=1)
            test_correct += int((pred == label).sum())
            test_loss += loss.item()

            num_test_nodes = batch.test_mask.sum().item()
            test_correct /= num_test_nodes
            # test_loss /= num_test_nodes
            test_accuracies.append(test_correct)
            test_losses.append(test_loss)

    average_test_loss = np.mean(test_losses)
    accuracy_test = np.mean(test_accuracies)
    return average_test_loss, accuracy_test

def validationPhase(model: Module, criterion: CrossEntropyLoss, val_loader: DataLoader, metricsTracker: MetricTracker, device):
    model.eval()
    val_accuracies = []
    val_losses = []
    with no_grad():
        for batch in val_loader:
            val_loss = 0.0
            val_correct = 0.0
            # Similar to the training loop, perform forward pass, compute loss, and accuracy
            # batch = batch.to(device)

            outputs, h = model(batch.x, batch.edge_index)
            loss = criterion(outputs[batch.val_mask], batch.y[batch.val_mask])

            metricsTracker.update(outputs[batch.val_mask], batch.y[batch.val_mask]) # pass prediction and label to metrics object to calculate metrics
            pred = outputs[batch.val_mask].argmax(dim=1)  # Use the class with highest probability.
            # correct_val += (pred[batch.val_mask] == batch.y[batch.val_mask]).sum()
            # total_val_loss += loss

            val_correct += int((pred == batch.y[batch.val_mask]).sum())
            val_loss += loss.item()

            # num_val_nodes = batch.val_mask.sum().item()
            # val_correct /= num_val_nodes
            # val_loss /= num_train_nodes
            val_accuracies.append(val_correct / batch.val_mask.sum().item())
            val_losses.append(val_loss)

    accuracy_val = np.mean(val_accuracies)
    average_val_loss = np.mean(val_losses)
    return average_val_loss, accuracy_val

def trainer(model: Module, criterion: CrossEntropyLoss, optimizer: Optimizer, trainLoader: DataLoader, device) -> float:
    model.train()
    accuracies_train: list[float] = []
    train_losses: list[float] = []
    for batch in trainLoader:  # Iterate in batches over the training dataset.
        train_loss = 0.0
        correct = 0.0
        # batch = batch.to(device)
        out, _ = model(batch.x, batch.edge_index)  # Perform a single forward pass.
        
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        pred = out[batch.train_mask].argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch.y[batch.train_mask]).sum())  # Check against ground-truth labels.
        train_loss += loss.item()
        num_train_nodes = batch.train_mask.sum().item()
        correct /= num_train_nodes
        accuracies_train.append(correct)
        train_losses.append(train_loss)

    average_accuracy_train = np.mean(accuracies_train)  # Derive ratio of correct predictions over nodes in gerah
    average_train_loss = np.mean(train_losses)
    return average_train_loss, average_accuracy_train

def trainPhase(train_loader: DataLoader, val_loader: DataLoader, num_classes: int, epochs: int, model: Module, criterion: CrossEntropyLoss, optimizer: Optimizer, device, save_experiment: bool = True, experiment_output_dir="./experiments"):
    metricCollection = MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="macro"),
        MulticlassROC(num_classes=num_classes, thresholds=None),
        MulticlassPrecisionRecallCurve(num_classes=num_classes, thresholds=None),
        MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None),
        MulticlassConfusionMatrix(num_classes=num_classes),
        MulticlassF1Score(num_classes=num_classes),
        MulticlassPrecision(num_classes=num_classes),
        MulticlassRecall(num_classes=num_classes)
    ])
    model.to(device)
    metricCollection.to(device)
    averageValLosses: list[float] = []
    averageTrainLosses: list[float] = []

    earlyStopper = EarlyStopper(patience=3, min_delta_factor=0.01)

    for epoch in range(0, epochs):
        average_train_loss, accuracy_train = trainer(model, criterion, optimizer, train_loader, device)
        average_val_loss, accuracy_val = validationPhase(model, criterion, val_loader, metricCollection, device)
        averageTrainLosses.append(average_train_loss)
        averageValLosses.append(average_val_loss)
        trainAverageAccStr = f"average over each label Acc: {accuracy_train:.4f}"
        trainLossStr = f"Train Loss: {average_train_loss:.4f}"
        valLossStr = f'Validation Loss: {average_val_loss:.4f}'
        valAccuracyStr = f'Validation Accuracy: {accuracy_val:.4f}'
        print(f"Epoch: {(epoch+1):03d}/{epochs}: {trainAverageAccStr}, {trainLossStr}, {valLossStr}, {valAccuracyStr}")
        # while loop it?
        if earlyStopper.early_stop(average_val_loss):
            break

    if save_experiment:
        saveExperiment(
            output_dir=experiment_output_dir,
            averageTrainLoesses=averageTrainLosses,
            averageValLosses=averageValLosses,
            metricCollection=metricCollection
        )

def save_output_graph(hidden_embeddings: torch.Tensor, edge_index: torch.Tensor, labels: torch.Tensor):
    graph_ptg = Data(x=hidden_embeddings, edge_index=edge_index, y=labels)
    graph_nx = to_networkx(graph_ptg)
    for node, node_feature in enumerate(hidden_embeddings):
        graph_nx.nodes[node].update(node_feature) 
    graph_nx.graph["y"] = labels
    date_time_str = get_now_datetime()
    GRAPHML_FILENAME = f"hidden_embedding_graph{date_time_str}"
    nx.write_graphml(graph_nx, f"{GRAPHML_FILENAME}.graphml")
