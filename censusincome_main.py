import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb
from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec
from multitaskrec.train import MPTRecTrainManager


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load dataset
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=args.seed
    )
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # build model
    model = MPTRec(
        num_tasks=2,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=127,
        expert_dnn_hidden_units=[256, 128],
        tower_dnn_hidden_units=[64, 32],
        reg_embedding=0.006,
        reg_dnn=3e-5,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    # build train manager
    train_manager = MPTRecTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=["Income", "Marital"],
        lr=1e-3,
        batch_size=256,
        epochs=30,
        patience=5,
        gen_coe=0.9,
        env_coe=0.1,
        clustering_interval=2,
        wandb_log=args.wandb_log,
    )

    # counting parameters and floating-point operands
    train_manager.compute_cost()

    # training
    if args.wandb_log:
        wandb.init(
            project="MULTITASKREC",
            config={
                "model": "MPTRec",
                "dataset": "CensusIncome",
                "seed": args.seed,
            },
        )
        train_manager.train()
    else:
        train_manager.train()

    # testing
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader)
    print(
        "AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}".format(
            auc_test[0], auc_test[1]
        )
    )
    if args.wandb_log:
        wandb.log({"AUC-Test-Income": auc_test[0], "AUC-Test-Marital": auc_test[1]})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1685480945)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wandb_log", type=bool, default=False)
    args = parser.parse_args()

    main(args)
