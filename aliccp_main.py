import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec
from multitaskrec.train import MPTRecTrainManager


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load dataset
    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 10000000)
    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 1000000)
    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 10000000)
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    # build model
    model = MPTRec(
        num_tasks=2,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=90,
        expert_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        dropout=[0.1, 0.3],
        reg_embedding=1e-4,
        reg_dnn=7e-6,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    # build train manager
    train_manager = MPTRecTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['CTR', 'CVR'],
        lr=1e-4,
        batch_size=2000,
        epochs=3,
        patience=3,
        gen_coe=0.9,
        env_coe=0.1,
        clustering_interval=2,
    )

    # counting parameters and floating-point operands
    train_manager.compute_cost()

    # training
    train_manager.train(way=args.way)

    # testing
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader, way=args.way)
    print(
        "AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}".format(
            auc_test[0], auc_test[1]
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--way", type=str, default="all")
    args = parser.parse_args()

    main(args)
