import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import STEM
from multitaskrec.train import TrainManager


def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load data
    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 100000)
    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 10000)
    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 100000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    # load model
    AliCCP_Vocabulary_Size.pop("101")  # 干扰特征
    AliCCP_Vocabulary_Size.pop("301")
    device = torch.device(f"cuda:{args.gpu}")
    model = STEM(
        task_num=1,
        shared_expert_num=1,
        specific_expert_num=1,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=80,
        expert_dnn_hidden_unit=[128, 64],
        tower_dnn_hidden_unit=[32, 32],
        reg_embedding=1e-6,
        reg_dnn=1e-6,
    )
    model.to(device)
    
    weight = torch.load(f'stem_alicpp_{args.seed}.pt')
    shared_weight = OrderedDict((n, weight[n]) for n in weight if "shared" in n)

    model.load_state_dict(shared_weight, strict=False)
    model.freeze_params()

    frozen_params_num = 0
    for n in shared_weight:
        frozen_params_num += shared_weight[n].numel()
    print(f'frozen_params:{frozen_params_num}')

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['CTR', 'CVR', 'ESI'],
        lr=1e-4
    )

    train_manager.compute_cost()
    
    train_manager.train_one_task(2)
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_one_task(test_loader, 2)
    print('AUC-Test-ESI:{:.4f}'.format(auc_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    # 1688723512, 1688723740, 1688738016
    parser.add_argument("--seed", type=int, default=1688723512)
    args = parser.parse_args()
 
    main(args)
