import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append('/home/hl/MultiTask/')

from utils.models import SingleTask
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

    model = SingleTask(
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=127,
        shared_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=0,
        reg_dnn=0,
    )
    device = torch.device("cuda:3")
    model.to(device)

    from utils.functions import compute_cost_0
    compute_cost_0(model, train_loader)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Income', 'Marital'],
        lr=1e-3
    )
    train_manager.train(1, task_id)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader, 1, task_id)
    task_name = ['Income', 'Marital']
    print('AUC-Test-{}:{:.4f}'.format(task_name[task_id], auc_test[0]))


if __name__ == '__main__':
    for task_id in range(0, 2):
        for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
            main()
