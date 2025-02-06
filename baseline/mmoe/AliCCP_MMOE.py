import argparse
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MMOE
from multitaskrec.train import TrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = MMOE(
        num_tasks=3,
        num_experts=3,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=80,
        expert_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32),
        reg_embedding=1e-6,
        reg_dnn=1e-6,
        dropout=(0.1, 0.3),
    )
    device = torch.device(f"cuda:{gpu}")
    model.to(device)
    
    # from utils.functions import compute_cost_0
    # compute_cost_0(model, train_loader)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['CTR', 'CVR', 'BSI'],
        lr=1e-4
    )
    train_manager.train_multi_task(3)
   
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, 3)
    print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}, AUC-Test-BSI:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    train_dataset = AliCCPDataset('/home/huangle/MultiTask/dataset/AliCCP/ctr_cvr.train', 10000000)
    val_dataset = AliCCPDataset('/home/huangle/MultiTask/dataset/AliCCP/ctr_cvr.dev', 1000000)
    test_dataset = AliCCPDataset('/home/huangle/MultiTask/dataset/AliCCP/ctr_cvr.test', 10000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    # for seed in [1688723512, 1688723740, 1688738016, 1688749593, 1688762746]:
    #     main()

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1688723512)

    args = parser.parse_args()
    gpu = args.gpu
    seed = args.seed
    print(f'gpu:{gpu}, seed:{seed}')
    main()
    