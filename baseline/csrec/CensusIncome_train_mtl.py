import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append('/home/hl/MultiTask/')

from utils.models import SparseSharing
from utils.train import CsRecTrainManager
from utils.dataset import CensusIncomeDataset
from utils.config import CensusIncome_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/train.gz')
    test_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
  
    model = SparseSharing(
        num_tasks=2,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=127,
        shared_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=3e-4,
    )
    device = torch.device("cuda:3")
    model.to(device)

    all_mask = []
    for i in range(2):
        all_mask.append(
            torch.load(f'/home/hl/MultiTask/baseline/csrec/CensusIncome/two_task/mask_{seed}_{i}.pt'))
   
    from fvcore.nn import FlopCountAnalysis
    from utils.functions import count_params
    count_params(model)
    for name in all_mask[0]:
        a = (1 - all_mask[0][name]) * (1 - all_mask[1][name])
        print('No training required:', a.sum())
    for _, _, features in train_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        flops = FlopCountAnalysis(model, features)
        print('FLOPs:', float(flops.total() / 1e6))
        break
    
    train_manager = CsRecTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        all_mask=all_mask,
        task_name=['Income', 'Marital'],
        lr=1e-3
    )
    train_manager.train(2)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader, 2)
    print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))


if __name__ == '__main__':
    for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
        main()
