import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import AliCCP_Vocabulary_Size
from utils.dataset import AliCCPDataset
from utils.models import MPTRec
from utils.train import MPTRecTrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    device = torch.device(f"cuda:{gpu}")
    mptrec = MPTRec(
        num_tasks=2,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=80,
        expert_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        dropout=[0.1, 0.3],
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device
    )
    mptrec.to(device)
    mptrec.base_network.load_state_dict(torch.load('/home/huangle/MultiTask/ali_base.pt'))
    mptrec.embedding_networks.load_state_dict(torch.load('/home/huangle/MultiTask/ali_embedding.pt'))

    # from utils.functions import compute_cost_0
    # compute_cost_0(mptrec, train_loader)

    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['CTR', 'CVR', 'BSI'],
        lr=1e-4,
        batch_size=2000,
        uni_coe=uni_coe,
        env_coe=env_coe
    )
    train_manager.train_two_task()

    mptrec.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_two_task(test_loader)
    print(f'AUC-Test-CTR:{auc_test[0]:.4f}, AUC-Test-CVR:{auc_test[1]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1688723512)

    train_dataset = AliCCPDataset('/home/huangle/MultiTask/dataset/AliCCP/ctr_cvr.train', 10000000)
    val_dataset = AliCCPDataset('/home/huangle/MultiTask/dataset/AliCCP/ctr_cvr.dev', 1000000)
    test_dataset = AliCCPDataset('/home/huangle/MultiTask/dataset/AliCCP/ctr_cvr.test', 10000000)
    
    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.0001
    reg_dnn = 7e-6

    # for seed in [1688723512, 1688723740, 1688738016, 1688749593, 1688762746]:
    #     main()

    args = parser.parse_args()
    gpu = args.gpu
    seed = args.seed
    print(f'gpu:{gpu}, seed:{seed}')
    main()
