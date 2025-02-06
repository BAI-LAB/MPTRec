import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('')

from config import ByteRec_Vocabulary_Size
from multitaskrec.dataset import ByteRecDataset
from multitaskrec.model import PLE
from multitaskrec.train import TrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = ByteRecDataset('data/ByteRec/train.gz')
    val_dataset = ByteRecDataset('data/ByteRec/val.gz')
    test_dataset = ByteRecDataset('data/ByteRec/test.gz')
    train_loader = DataLoader(train_dataset, batch_size=4000)
    val_loader = DataLoader(val_dataset, batch_size=4000)
    test_loader = DataLoader(test_dataset, batch_size=4000)
    
    task_num = 2
    model = PLE(
        num_tasks=task_num,
        input_size=32,
        feature_vocabulary=ByteRec_Vocabulary_Size,
        embedding_size=4,
        shared_expert_num=1,
        specific_expert_num=1,
        num_levels=2,
        expert_dnn_hidden_units=(128, ),
        tower_dnn_hidden_units=(32, 32),
        reg_embedding=1e-6,
        reg_dnn=1e-6
    )
    device = torch.device("cuda:5")
    model.to(device)

    # from utils.functions import compute_cost_0
    # compute_cost_0(model, train_loader)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Finish', 'Like', 'Duration_time'],
        epochs=10,
        lr=1e-4,
    )
    train_manager.train_multi_task(task_num)

    model.load_state_dict(train_manager.best_weight)
    torch.save(train_manager.best_weight, f'baseline/ple/ByteRec_{seed}.pt')
    auc_test = train_manager.evaluation_multi_task(test_loader, task_num)
    if task_num == 2:
        print('AUC-Test-Finish:{:.4f}, AUC-Test-Like:{:.4f}'.format(auc_test[0], auc_test[1]))
    else:
        print('AUC-Test-Finish:{:.4f}, AUC-Test-Like:{:.4f}, AUC-Test-Duration_time:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    for seed in [1688723512]:
        main()
