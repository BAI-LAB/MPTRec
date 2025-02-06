import sys
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append('')

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import SparseSharing
from multitaskrec.train import CsRecTrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset('data/CensusIncome/#train.gz')
    test_dataset = CensusIncomeDataset('data/CensusIncome/#test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
  
    model = SparseSharing(
        num_tasks=3,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        shared_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=3e-4,
    )
    device = torch.device("cuda:3")
    model.to(device)

    all_mask = []
    for i in range(3):
        all_mask.append(
            torch.load(f'baseline/csrec/CensusIncome/three_task/mask_{seed}_{i}.pt'))

    from fvcore.nn import FlopCountAnalysis
    from utils.functions import count_params
    count_params(model)
    for name in all_mask[0]:
        a = (1 - all_mask[0][name]) * (1 - all_mask[1][name])
        print('No training required:', a.sum())
    for _, _, _, features in train_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        flops = FlopCountAnalysis(model, features)
        print('FLOPs:', flops.total())
        break

    train_manager = CsRecTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        all_mask=all_mask,
        task_name=['Income', 'Marital', 'Education'],
        lr=1e-3
    )
    train_manager.train_multi_task(3)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, 3)
    print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}, AUC-Test-Education:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
        main()
        break
