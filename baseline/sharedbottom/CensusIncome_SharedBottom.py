import sys
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append('')

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import SharedBottom
from multitaskrec.train import TrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset('data/CensusIncome/train.gz')
    test_dataset = CensusIncomeDataset('data/CensusIncome/test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    task_num = 2
    model = SharedBottom(
        num_tasks=task_num,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        shared_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=3e-4,
        reg_dnn=3e-4
    )
    device = torch.device("cuda:3")
    model.to(device)
    
    # from utils.functions import compute_cost_0
    # compute_cost_0(model, train_loader)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Income', 'Marital', 'Sex'],
        lr=1e-3
    )
    train_manager.train_multi_task(task_num)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, task_num)
    if task_num == 2:
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))
    else:
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}, AUC-Test-Sex:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
        main()

    print('full-training')
