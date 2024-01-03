import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.models import MPTRec
from utils.train import MPTRecTrainManager
from utils.dataset import CensusIncomeDataset
from utils.config import CensusIncome_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 加载数据集
    train_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/train.gz')
    test_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    env_ids = torch.load('/home/hl/MultiTask/data/CensusIncome/#env_id.gz')
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # 设置模型
    task_num = 2
    device = torch.device("cuda:1")
    mptrec = MPTRec(
        num_tasks=task_num,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device
    )
    mptrec.to(device)

    # from utils.functions import compute_cost_0
    # compute_cost_0(mptrec, train_loader)

    # 设置训练器
    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['Income', 'Marital', 'Sex'],
        lr=1e-3,
        batch_size=256,
        uni_coe=uni_coe,
        env_coe=env_coe
    )

    if task_num == 2:
        train_manager.train_two_task()
        mptrec.load_state_dict(train_manager.best_weight)
        auc_test = train_manager.evaluation_two_task(test_loader)
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))
    else:
        train_manager.train_three_task()
        mptrec.load_state_dict(train_manager.best_weight)
        auc_test = train_manager.evaluation_three_task(test_loader)
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}, AUC-Test-Sex:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.006
    reg_dnn = 3e-5
    for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
        main()
