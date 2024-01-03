import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('/home/hl/MultiTask/')

from utils.models import SparseSharing
from utils.dataset import AliCCPDataset
from utils.config import AliCCP_Vocabulary_Size
from utils.train import SparseSharingTrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = SparseSharing(
        num_tasks=3,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=80,
        shared_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32),
        reg_embedding=1e-6
    )
    device = torch.device("cuda:2")
    model.to(device)
  
    all_mask = []
    for i in range(3):
        all_mask.append(torch.load(f'/home/hl/MultiTask/baseline/csrec/AliCpp/three_task/mask_{seed}_{i}.pt'))
    
    from utils.functions import compute_cost_1
    compute_cost_1(model, all_mask, train_loader)
    
    train_manager = SparseSharingTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        all_mask=all_mask,
        task_name=['CTR', 'CVR', 'BSI'],
        lr=1e-4,
    )
    train_manager.train_multi_task(3)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, 3)
    print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}, AUC-Test-BSI:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    train_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCCP/ctr_cvr.train', 100000)
    val_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCCP/ctr_cvr.dev', 10000)
    test_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCCP/ctr_cvr.test', 100000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    for seed in [1688723512, 1688723740, 1688738016, 1688749593, 1688762746]:
        main()
