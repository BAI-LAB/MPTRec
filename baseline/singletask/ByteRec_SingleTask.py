import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('/data/hl/MultiTask/')

from multitaskrec.model import SingleTask
from multitaskrec.train import TrainManager
from multitaskrec.dataset import ByteRecDataset
from config import ByteRec_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = SingleTask(
        feature_vocabulary=ByteRec_Vocabulary_Size,
        embedding_size=4,
        input_size=32,
        shared_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32),
        reg_embedding=0,
        reg_dnn=0,
    )
    device = torch.device("cuda:2")
    model.to(device)
    
    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['Finish', 'Like'],
        epochs=10,
        lr=1e-4
    )
    train_manager.train(1, task_id)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader, 1, task_id)
    task_name = ['Finish', 'Like']
    print('AUC-Test-{}:{:.4f}'.format(task_name[task_id], auc_test[0]))


if __name__ == '__main__':
    train_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/train.gz')
    val_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/val.gz')
    test_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/test.gz')
    train_loader = DataLoader(train_dataset, batch_size=4000)
    val_loader = DataLoader(val_dataset, batch_size=4000)
    test_loader = DataLoader(test_dataset, batch_size=4000)

    for task_id in range(2):
        for seed in [1688723512, 1688723740, 1688738016]:
            main()
