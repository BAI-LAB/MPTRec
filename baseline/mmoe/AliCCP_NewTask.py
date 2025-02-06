import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('')

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

    reg_embedding = 1e-6
    reg_dnn = 1e-6

    model = MMOE(
        num_tasks=1,
        num_experts=3,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=80,
        expert_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        dropout=(0.1, 0.3),
    )
    weight = torch.load(f'baseline/mmoe/alicpp_{seed}.pt')
    param_names = [n for n in weight]
    trainable_params = 0
    for name in param_names:
        if 'tower_networks' in name or 'gate_networks' in name:
            trainable_params += weight[name].numel()
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
        task_name=['CTR', 'CVR', 'ESI'],
        lr=1e-4
    )
    train_manager.train_one_task(2)
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_one_task(test_loader, 2)
    print('AUC-Test-ESI:{:.4f}'.format(auc_test))


if __name__ == '__main__':
    train_dataset = AliCCPDataset('data/AliCCP/ctr_cvr.train', 100000)
    val_dataset = AliCCPDataset('data/AliCCP/ctr_cvr.dev', 10000)
    test_dataset = AliCCPDataset('data/AliCCP/ctr_cvr.test', 100000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)
    
    for seed in [1688723512]:
        main()
