import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multitaskrec.model import MPTRec
from multitaskrec.train import MPTRecTrainManager
from multitaskrec.dataset import CensusIncomeDataset
from config import CensusIncome_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/train.gz')
    test_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    device = torch.device("cuda:3")
    model = MPTRec(
        num_tasks=2,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=127,
        expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device
    )
    model.to(device)

    # from utils.functions import compute_cost_0
    # compute_cost_0(model, train_loader)

    train_manager = MPTRecTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['Income', 'Marital'],
        lr=1e-3,
        batch_size=256,
        uni_coe=uni_coe,
        env_coe=env_coe,
    )
    train_manager.train_two_task()

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_two_task(test_loader)
    print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))


if __name__ == '__main__':
    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.006
    reg_dnn = 3e-5
    for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
        main()
    print('两个任务CensusIncome')

    # for uni_coe in np.arange(0, 1.1, 0.1):
    #     env_coe = 1 - uni_coe
    #     reg_embedding = 0.006
    #     reg_dnn = 3e-5
    #     seed = 1685480945
    #     main()
    # print('寻找最佳的uni_coe和env_coe')
