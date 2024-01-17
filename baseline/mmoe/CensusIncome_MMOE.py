import argparse

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MMOE
from multitaskrec.train import TrainManager


def main(seed, gpu):
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load dataset
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # build model
    model = MMOE(
        task_num=2,
        expert_num=3,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=127,
        expert_dnn_hidden_unit=[256, 128],
        tower_dnn_hidden_unit=[64, 32],
        reg_embedding=3e-4,
        reg_dnn=3e-4,
    )
    device = torch.device(f"cuda:{gpu}")
    model.to(device)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=["Income", "Marital"],
        lr=1e-3,
        epochs=10,
        patience=1,
    )

    # counting parameters and floating-point operands
    train_manager.compute_cost()

    # training
    wandb.init(
        project="multitaskrec",
        config={"model": "mmoe", "dataset": "CensusIncome", "seed": seed},
    )
    train_manager.train()
    wandb.finish()

    # evaluation
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader)
    print(
        "AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}".format(
            auc_test[0], auc_test[1]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1685480945)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args.seed, args.gpu)
