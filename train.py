# %%
import torch
from pytorch_geometric_dataset import LOL_Dataset
# from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
# import numpy as np
from torch_geometric.transforms import NormalizeFeatures, Compose, ToDevice
# from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader
from train_utils import testPhase, trainPhase, CustomRandomNodeSplitMasker, CustomRandomNodeUnderSampler, ConcatNodeCentralities
from models import GCN

# %%
ROOT_DIR="PTG_data"
OUTPUT_DIR = "./experiments"
NUM_EPOCHS = 100
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 10
HIDDEN_CHANNELS = 32
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
nodeUnderSample = CustomRandomNodeUnderSampler(RANDOM_STATE, allowSelfLoops=False)
nodeSPlitMasker = CustomRandomNodeSplitMasker(
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE
)
concatNodeCentralitiesTransformer = ConcatNodeCentralities()
train_transformer = Compose(
    [
        # nodeUnderSample,
        NormalizeFeatures(),
        ToDevice(DEVICE),
        nodeSPlitMasker
    ]
)
train_val_test_pre_transforms = Compose(
    [
        nodeUnderSample,
        concatNodeCentralitiesTransformer
    ]
)
val_test_transformer = Compose(
    [
        # nodeUnderSample,
        ToDevice(DEVICE),
        nodeSPlitMasker
    ]
)

train_dataset = LOL_Dataset(
    root=ROOT_DIR,
    transform=train_transformer,
    pre_transform=train_val_test_pre_transforms
)
val_test_dataset = LOL_Dataset(
    root=ROOT_DIR,
    transform=val_test_transformer,
    pre_transform=train_val_test_pre_transforms
)
# %%
LOL_GRAPH = train_dataset[0]
DATA_SET_NUM_FEATURES = LOL_GRAPH.num_node_features
NUM_NODES = LOL_GRAPH.num_nodes
DATA_SET_CLASSES = train_dataset.num_classes
# LOL_GRAPH = train_dataset[0]
# %%
print(f"{DATA_SET_NUM_FEATURES = }")
print(f"{DATA_SET_CLASSES = }")
print(f"{NUM_NODES = }")
# %%
# splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# # Split the data indices into train and test sets
# for train_index, test_index in splitter.split(np.zeros(NUM_NODES), LOL_GRAPH.y):
#     train_indices, test_indices = train_index, test_index

# # Now, split the training indices into training and validation sets
# splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)  # Assuming 20% for validation set

# for train_index, val_index in splitter.split(LOL_GRAPH.y[train_indices], LOL_GRAPH.y[train_indices]):
#     train_indices, val_indices = train_index, val_index

# # %%
# train_mask = torch.zeros(LOL_GRAPH.num_nodes, dtype=torch.bool)
# # set indices where to true for nodes to include from indices in train_indices
# train_mask[train_indices] = True

# val_mask = torch.zeros(LOL_GRAPH.num_nodes, dtype=torch.bool)
# val_mask[val_indices] = True

# test_mask = torch.zeros(LOL_GRAPH.num_nodes, dtype=torch.bool)
# test_mask[test_indices] = True
# # %%

# %%
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)
# test_sampler = SubsetRandomSampler(test_indices)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# val_loader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
# test_loader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE)
# %%
gcn = GCN(DATA_SET_NUM_FEATURES, HIDDEN_CHANNELS, DATA_SET_CLASSES)
gcn_optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)
gcn_criterion = torch.nn.CrossEntropyLoss()
print(gcn)
# %%
trainPhase(
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=DATA_SET_CLASSES,
    epochs=NUM_EPOCHS,
    model=gcn,
    criterion=gcn_criterion,
    optimizer=gcn_optimizer,
    device=DEVICE,
    save_experiment=True,
    experiment_output_dir=OUTPUT_DIR
)

# %%[markdown]
#  ## Results

#  #### Model Test
# %%
test_loss, test_accuracy = testPhase(gcn, gcn_criterion, test_loader, DEVICE)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%')

# %%


# test_loss, test_accuracy = testPhase(model_W_UD, criterion_W_UD, test_loader, DEVICE)
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%')

# %%[markdown]

# ## Visualize the results

