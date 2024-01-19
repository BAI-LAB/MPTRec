import argparse
from collections import OrderedDict

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
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # load model
    CensusIncome_Vocabulary_Size.pop("education")
    model = STEM(
        task_num=1,
        shared_expert_num=1,
        specific_expert_num=1,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        expert_dnn_hidden_unit=[256, 128],
        tower_dnn_hidden_unit=[64, 32],
        reg_embedding=1e-4,
        reg_dnn=1e-4,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    weight = torch.load(f'stem_census_{args.seed}.pt')
    shared_weight = OrderedDict((n, weight[n]) for n in weight if "shared" in n)
    
    frozen_params_num = 0
    for n in shared_weight:
        frozen_params_num += shared_weight[n].numel()
    print(f'frozen_params:{frozen_params_num}')

    model.load_state_dict(shared_weight, strict=False)
    model.freeze_params()
    
    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Income', 'Marital', 'Education'],
        lr=1e-3
    )

    train_manager.compute_cost()

    train_manager.train_one_task(2)
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_one_task(test_loader, 2)
    print('AUC-Test-Education:{:.4f}'.format(auc_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    # 1685480945, 1685463909, 1685477428
    parser.add_argument("--seed", type=int, default=1685480945)
    args = parser.parse_args()
 
    main(args)
