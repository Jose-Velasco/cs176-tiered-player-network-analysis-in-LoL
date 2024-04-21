# %%
import torch
from pytorch_geometric_dataset import LOL_Dataset
from torch_geometric.transforms import NormalizeFeatures, Compose, ToDevice
from train_utils import testPhase, trainPhase, CustomRandomNodeSplitMasker, CustomRandomNodeUnderSampler,ConcatNodeCentralities
# %%
ROOT_DIR="PTG_data"
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

# nodeUnderSample = CustomRandomNodeUnderSampler(RANDOM_STATE)
# nodeSPlitMasker = CustomRandomNodeSplitMasker(
#     test_size=TEST_SIZE,
#     validation_size=VALIDATION_SIZE,
#     random_state=RANDOM_STATE
# )
train_transformer = Compose(
    [
        NormalizeFeatures(),
        # ToDevice(DEVICE),
        # nodeUnderSample,
        # nodeSPlitMasker,
        # ConcatNodeCentralities()
    ]
)
# %%

train_dataset = LOL_Dataset(
    root=ROOT_DIR,
    transform=train_transformer,
    pre_transform=ConcatNodeCentralities()
)

# %%
train_dataset[0]
# %%
train_dataset[0].num_nodes
train_dataset[0].num_node_features

# %%
