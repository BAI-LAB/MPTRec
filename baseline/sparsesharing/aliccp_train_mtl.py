import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import SparseSharing
from multitaskrec.train import SparseSharingTrainManager


def main():
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load dataset
    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 10000000)
    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 1000000)
    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 10000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    # build model
    model = SparseSharing(
        num_tasks=2,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=90,
        shared_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        reg_embedding=1e-6,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    all_mask = []
    for i in range(2):
        all_mask.append(torch.load(f'/home/hl/MultiTask/baseline/csrec/AliCpp/two_task/mask_{seed}_{i}.pt'))

    train_manager = SparseSharingTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        all_mask=all_mask,
        task_name=['CTR', 'CVR'],
        lr=1e-4,
        epochs=10,
        patience=3,
        wandb_log=args.wandb_log,
    )
    train_manager.train(2)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader, 2)
    print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}'.format(auc_test[0], auc_test[1]))


if __name__ == '__main__':
    train_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.train', 10000000)
    val_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.dev', 1000000)
    test_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.test', 10000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    for seed in [1688723512, 1688723740, 1688738016, 1688749593, 1688762746]:
        main()
