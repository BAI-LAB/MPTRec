#%%
import sys
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append('/home/hl/MultiTask')
from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import PLE, MPTRec

warnings.filterwarnings('ignore')

seed = 1685480945
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

train_dataset = CensusIncomeDataset('data/CensusIncome/train.gz')
test_dataset = CensusIncomeDataset('data/CensusIncome/test.gz')
val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
env_ids = torch.load('data/CensusIncome/#env_id.gz')
train_loader = DataLoader(train_dataset, batch_size=2000)
val_loader = DataLoader(val_dataset, batch_size=2000)
test_loader = DataLoader(test_dataset, batch_size=2000)


# %% 选择模型MPTRec
task_num = 2
device = torch.device("cuda:1")
model = MPTRec(
    num_tasks=task_num,
    feature_vocabulary=CensusIncome_Vocabulary_Size,
    embedding_size=4,
    input_size=123,
    expert_dnn_hidden_units=(256, 128),
    tower_dnn_hidden_units=(64, 32),
    reg_embedding=0.006,
    reg_dnn=1e-5,
    device=device
)
model.to(device)


# %% 选择模型PLE
# model = PLE(
#     num_tasks=2,
#     input_size=123,
#     feature_vocabulary=CensusIncome_Vocabulary_Size,
#     embedding_size=4,
#     shared_expert_num=1,
#     specific_expert_num=1,
#     num_levels=2,
#     expert_dnn_hidden_units=(256, ),
#     tower_dnn_hidden_units=(64, 32),
#     reg_embedding=3e-4,
#     reg_dnn=3e-4
# )
# model.load_state_dict(torch.load(f'baseline/ple/CensusIncome_{seed}.pt'))
# model.to(device)

# %% 得到通用表征和专有表征
generic_rep_list, proprietary_rep_0_list, proprietary_rep_1_list = [], [], []
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    generic_rep, proprietary_reps = model.get_reps(features)
    generic_rep_list.append(generic_rep)
    proprietary_rep_0_list.append(proprietary_reps[0])
    proprietary_rep_1_list.append(proprietary_reps[1])
generic_rep = torch.cat(generic_rep_list).detach().cpu()
proprietary_rep_0 = torch.cat(proprietary_rep_0_list).detach().cpu()
proprietary_rep_1 = torch.cat(proprietary_rep_1_list).detach().cpu()

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

#%%
color = ['#008000', '#0066CC', '#FF6666']
import numpy as np

inx = np.random.randint(0, 3, size=40000)
# %%
ccc = []
for i in inx:
    ccc.append(color[i])
# %%
ccc
len(embeddings_2d)
# %%
plt.scatter(embeddings_2d[:point_size, 0], embeddings_2d[:point_size, 1], c=ccc, s=1)

plt.title(f'{point_size} points, before training', font={'family':'Arial', 'size':10})
plt.rcParams.update({'font.size':10, 'font.weight':'bold'})

# %% 两类表征的二维可视化
# start_point = 0
# point_size = 20000
# rep_embeddings = torch.cat([generic_rep[start_point:start_point+point_size], proprietary_rep_0[start_point:start_point+point_size], proprietary_rep_1[start_point:start_point+point_size]])

# tsne = manifold.TSNE(n_components=2, random_state=2023)
# embeddings_2d = tsne.fit_transform(rep_embeddings)

# plt.scatter(embeddings_2d[:point_size, 0], embeddings_2d[:point_size, 1], c='green', s=1)
# plt.scatter(embeddings_2d[point_size:2*point_size, 0], embeddings_2d[point_size:2*point_size, 1], c='#FF6666', s=1)
# plt.scatter(embeddings_2d[2*point_size:3*point_size, 0], embeddings_2d[2*point_size:3*point_size, 1], c='#0066CC', s=1)
# plt.title(f'{point_size} points, before training', font={'family':'Arial', 'size':10})
# plt.rcParams.update({'font.size':10, 'font.weight':'bold'})

# %%
