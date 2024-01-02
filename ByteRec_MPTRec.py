import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader
from utils.models import MPTRec
from utils.dataset import ByteRecDataset
from utils.train import MPTRecTrainManager
from utils.config import ByteRec_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=4000)
    val_loader = DataLoader(val_dataset, batch_size=4000)
    test_loader = DataLoader(test_dataset, batch_size=4000)
    env_ids = torch.load('/data/hl/MultiTask/data/ByteRec/env_id.gz')

    device = torch.device("cuda:5")
    mptrec = MPTRec(
        num_tasks=2,
        feature_vocabulary=ByteRec_Vocabulary_Size,
        embedding_size=4,
        input_size=32,
        expert_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device
    )
    mptrec.to(device)

    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['Finish', 'Like'],
        lr=1e-4,
        batch_size=4000,
        epochs=10, 
        uni_coe=uni_coe,
        env_coe=env_coe
    )
    train_manager.train_two_task()

    mptrec.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_two_task(test_loader)
    print('AUC-Test-Finish:{:.4f}, AUC-Test-Like:{:.4f}'.format(auc_test[0], auc_test[1]))


if __name__ == '__main__':
    train_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/train.gz')
    val_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/val.gz')
    test_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/test.gz')

    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.0001
    reg_dnn = 7e-6

    for seed in [1688723512, 1688723740, 1688738016]:
        main()
