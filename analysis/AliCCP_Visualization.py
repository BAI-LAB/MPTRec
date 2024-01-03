#%%
import sys
import torch
import warnings
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn import manifold
sys.path.append('/home/hl/MultiTask')
from utils.models import PLE, MPTRec
from utils.dataset import AliCCPDataset
from utils.config import AliCCP_Vocabulary_Size
warnings.filterwarnings('ignore')

seed = 1688723740
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

train_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCCP/ctr_cvr.train', 1000000)
val_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCCP/ctr_cvr.dev', 100000)
test_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCCP/ctr_cvr.test', 1000000)
train_loader = DataLoader(train_dataset, batch_size=2000)
val_loader = DataLoader(val_dataset, batch_size=2000)
test_loader = DataLoader(test_dataset, batch_size=2000)
env_ids = torch.load('/home/hl/MultiTask/data/AliCCP/env_id.gz')[:len(train_dataset)]
device = torch.device("cuda:7")

# %% 选择模型MPTRec
uni_coe = 0.9
env_coe = 0.1
reg_embedding = 0.0001
reg_dnn = 7e-6
model = MPTRec(
    num_tasks=2,
    feature_vocabulary=AliCCP_Vocabulary_Size,
    embedding_size=5,
    input_size=80,
    expert_dnn_hidden_units=(128, 64),
    tower_dnn_hidden_units=(32, 32),
    dropout=(0.1, 0.3),
    reg_embedding=reg_embedding,
    reg_dnn=reg_dnn,
    device=device
)
# model.load_state_dict(torch.load(f'/home/hl/MultiTask/AliCCP_{seed}.pt'))
model.to(device)

# %% 选择模型PLE
model = PLE(
    num_tasks=2,
    input_size=80,
    feature_vocabulary=AliCCP_Vocabulary_Size,
    embedding_size=5,
    shared_expert_num=1,
    specific_expert_num=1,
    num_levels=2,
    expert_dnn_hidden_units=(128, ),
    tower_dnn_hidden_units=(32, 32),
    reg_embedding=1e-6,
    reg_dnn=1e-6,
    dropout=(0.1, 0.3),
)
model.load_state_dict(torch.load(f'/home/hl/MultiTask/baseline/ple/AliCCP_{seed}.pt'))
model.to(device)

from utils.train import TrainManager
train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['CTR', 'CVR', 'BSI'],
        epochs=30,
        lr=1e-4,
    )
train_manager.train_multi_task(2)

# %% 得到通用表征和专有表征
count = 0
generic_rep_list, proprietary_rep_0_list, proprietary_rep_1_list = [], [], []
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    generic_rep, proprietary_reps = model.get_reps(features)
    generic_rep_list.append(generic_rep)
    proprietary_rep_0_list.append(proprietary_reps[0])
    proprietary_rep_1_list.append(proprietary_reps[1])
    count += 1
    if count == 100:
        break
generic_rep = torch.cat(generic_rep_list).detach().cpu()
proprietary_rep_0 = torch.cat(proprietary_rep_0_list).detach().cpu()
proprietary_rep_1 = torch.cat(proprietary_rep_1_list).detach().cpu()

# %% 两类表征的二维可视化
start_point = 0
point_size = 40000
rep_embeddings = torch.cat([generic_rep[start_point:start_point+point_size], proprietary_rep_0[start_point:start_point+point_size], proprietary_rep_1[start_point:start_point+point_size]])

tsne = manifold.TSNE(n_components=2, random_state=2023)
embeddings_2d = tsne.fit_transform(rep_embeddings)

plt.scatter(embeddings_2d[:point_size, 0], embeddings_2d[:point_size, 1], c='green', s=1)
plt.scatter(embeddings_2d[point_size:2*point_size, 0], embeddings_2d[point_size:2*point_size, 1], c='#FF6666', s=1)
plt.scatter(embeddings_2d[2*point_size:3*point_size, 0], embeddings_2d[2*point_size:3*point_size, 1], c='#0066CC', s=1)
plt.title(f'{point_size} points, after training', font={'size':10})
plt.rcParams.update({'font.size':10, 'font.weight':'bold'})

# %% 得到dnn_input并可视化
dnn_input_list = []
count = 0
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    dnn_input, _, _, _ = model.get_infos(features)
    dnn_input_list.append(dnn_input)
    count += 1
    if count == 20:
        break
dnn_input = torch.cat(dnn_input_list).detach().cpu()

start_point = 0
point_size = 40000

tsne = manifold.TSNE(n_components=2, random_state=2023)
embeddings_2d = tsne.fit_transform(dnn_input)

# plt.scatter(embeddings_2d[:point_size, 0], embeddings_2d[:point_size, 1], c='orange', s=1)
plt.scatter(embeddings_2d[:13333, 0], embeddings_2d[:13333, 1], c='green', s=1)
plt.scatter(embeddings_2d[13333:26666, 0], embeddings_2d[13333:26666, 1], c='#FF6666', s=1)
plt.scatter(embeddings_2d[26666:, 0], embeddings_2d[26666:, 1], c='#0066CC', s=1)
plt.title(f'{point_size} points, before training', font={'size':10})
plt.rcParams.update({'font.size':10, 'font.weight':'bold'})

# %%
