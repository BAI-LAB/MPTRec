import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import SingleTask
from multitaskrec.train import SingleTaskTrainManager


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
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    # build model
    model = SingleTask(
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=90,
        shared_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        reg_embedding=0,
        reg_dnn=0,
        dropout=[0.1, 0.3],
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    # build train manager
    task_names = ["CTR", "CVR"]
    train_manager = SingleTaskTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_id=args.task_id,
        task_name=task_names[args.task_id],
        lr=1e-4,
        epochs=10,
        patience=3,
    )

    # counting parameters and floating-point operands
    train_manager.compute_cost()

    # training
    train_manager.train()

    # testing
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader)
    print("AUC-Test-{}:{:.4f}".format(task_name[task_id], auc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args)
