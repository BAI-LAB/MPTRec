import argparse
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.config import CensusIncome_Vocabulary_Size
from utils.dataset import CensusIncomeDataset
from utils.models import MPTRec
from utils.train import MPTRecTrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset('/home/huangle/MultiTask/dataset/CensusIncome/#train.gz')
    test_dataset = CensusIncomeDataset('/home/huangle/MultiTask/dataset/CensusIncome/#test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    env_ids = torch.load('/home/huangle/MultiTask/dataset/CensusIncome/#env_id.gz')
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    device = torch.device(f"cuda:{gpu}")
    mptrec = MPTRec(
        num_tasks=2,
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
    mptrec.base_network.load_state_dict(torch.load('/home/huangle/MultiTask/ci_base.pt'))
    mptrec.embedding_networks.load_state_dict(torch.load('/home/huangle/MultiTask/ci_embedding.pt'))

    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['Income', 'Marital', 'Education'],
        lr=1e-3,
        batch_size=256,
        uni_coe=uni_coe,
        env_coe=env_coe
    )
    train_manager.train_two_task()
    mptrec.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_two_task(test_loader)
    print(f'AUC-Test-Income:{auc_test[0]:.4f}, AUC-Test-Marital:{auc_test[1]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1685480945)

    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.006
    reg_dnn = 3e-5

    # for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
    #     main()
    args = parser.parse_args()
    gpu = args.gpu
    seed = args.seed
    print(f'gpu:{gpu}, seed:{seed}')
    main()
