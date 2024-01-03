import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append('/home/hl/MultiTask/')

from utils.train import TrainManager
from utils.models import SharedBottom
from utils.dataset import CensusIncomeDataset
from utils.config import CensusIncome_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    reg_embedding = 3e-4
    reg_dnn = 3e-4

    train_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/#train.gz')
    test_dataset = CensusIncomeDataset('/home/hl/MultiTask/data/CensusIncome/#test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = SharedBottom(
        num_tasks=1,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        shared_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn
    )

    weight = torch.load(f'/home/hl/MultiTask/baseline/sharedbottom/census_income_{seed}.pt')
    param_names = [n for n in weight]
    trainable_params = 0
    for name in param_names:
        if 'tower_networks' in name:
            trainable_params += weight[name].numel()
            print(weight[name].shape)
            del weight[name]
    print(f'trainable_params:{trainable_params}')
    model.load_state_dict(weight, strict=False)
    device = torch.device("cuda:5")
    model.to(device)
    model.freeze_params()
    
    from fvcore.nn import FlopCountAnalysis
    for _, _, _, features in train_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        flops = FlopCountAnalysis(model, features)
        print(flops.by_module())
        break
    
    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Income', 'Marital', 'Education'],
        lr=1e-3
    )
    train_manager.train_one_task(2)
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_one_task(test_loader, 2)
    print('AUC-Test-Education:{:.4f}'.format(auc_test))


if __name__ == '__main__':
    for seed in [1685463909]:
        main()
