import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader


from multitaskrec.train import TrainManager
from multitaskrec.model import SharedBottom
from multitaskrec.dataset import ByteRecDataset
from config import ByteRec_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
        
    model = SharedBottom(
        num_tasks=2,
        feature_vocabulary=ByteRec_Vocabulary_Size,
        embedding_size=4,
        input_size=32,
        shared_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        reg_embedding=1e-6,
        reg_dnn=1e-6,
    )
    device = torch.device("cuda:1")
    model.to(device)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Finish', 'Like'],
        epochs=10,
        lr=1e-4,
        patience=5,
    )
    train_manager.train()

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader)
    print('AUC-Test-Finish:{:.4f}, AUC-Test-Like:{:.4f}'.format(auc_test[0], auc_test[1]))

    
if __name__ == '__main__':
    train_dataset = ByteRecDataset('dataset/Byte-Rec/train.gz')
    val_dataset = ByteRecDataset('dataset/Byte-Rec/val.gz')
    test_dataset = ByteRecDataset('dataset/Byte-Rec/test.gz')
    train_loader = DataLoader(train_dataset, batch_size=4000)
    val_loader = DataLoader(val_dataset, batch_size=4000)
    test_loader = DataLoader(test_dataset, batch_size=4000)

    for seed in [1688723512, 1688723740, 1688738016]:
        main()
