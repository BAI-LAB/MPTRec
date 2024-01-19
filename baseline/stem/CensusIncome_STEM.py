import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import STEM
from multitaskrec.train import TrainManager


def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load data
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", args.new_task)
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", args.new_task)
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # load model
    CensusIncome_Vocabulary_Size.pop(args.new_task)
    model = STEM(
        task_num=args.task_num,
        shared_expert_num=1,
        specific_expert_num=1,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        expert_dnn_hidden_unit=[256, 128],
        tower_dnn_hidden_unit=[64, 32],
        reg_embedding=3e-4,
        reg_dnn=3e-4,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Income', 'Marital', args.new_task],
        lr=1e-3,
        epochs=10,
    )

    # train_manager.compute_cost()

    train_manager.train_multi_task(args.task_num)
    # torch.save(train_manager.best_weight, f"stem_census_{args.seed}.pt")

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, args.task_num)
    if args.task_num == 2:
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))
    else:
        print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}, AUC-Test-{}:{:.4f}'.format(auc_test[0], auc_test[1], args.new_task, auc_test[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    # 1685480945, 1685463909, 1685477428
    parser.add_argument("--seed", type=int, default=1685480945)
    parser.add_argument("--new_task", type=str, default="education")
    parser.add_argument("--task_num", type=int, default=3)
    args = parser.parse_args()
 
    main(args)
    