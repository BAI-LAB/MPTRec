import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append('/home/hl/MultiTask/')

from utils.models import PLE
from utils.train import TrainManager
from utils.dataset import CensusIncomeDataset
from utils.config import CensusIncome_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/train.gz')
    test_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    task_num = 2
    model = PLE(
        num_tasks=task_num,
        input_size=123,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        shared_expert_num=1,
        specific_expert_num=1,
        num_levels=2,
        expert_dnn_hidden_units=(256, ),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=3e-4,
        reg_dnn=3e-4
    )
    device = torch.device("cuda:5")
    model.to(device)

    # from utils.functions import compute_cost_0
    # compute_cost_0(model, train_loader)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Income', 'Marital', 'Sex'],
        lr=1e-3,
    )
    train_manager.train_multi_task(task_num)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, task_num)
    if task_num == 2:
        # torch.save(train_manager.best_weight, f'/home/hl/MultiTask/baseline/ple/CensusIncome_{seed}.pt')
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))
    else:
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}, AUC-Test-Sex:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    for seed in [1685480945, 1685463909, 1685477428]:
        main()
