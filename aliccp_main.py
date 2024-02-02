import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec
from multitaskrec.train import MPTRecTrainManager


def main():
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.train', 10000000)
    val_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.dev', 1000000)
    test_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.test', 10000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))

    device = torch.device("cuda:0")
    model = MPTRec(
        num_tasks=2,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=90,
        expert_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        dropout=[0.1, 0.3],
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device
    )
    model.to(device)

    train_manager = MPTRecTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['CTR', 'CVR'],
        epochs=5,
        lr=1e-4,
        batch_size=2000,
        uni_coe=uni_coe,
        env_coe=env_coe
    )
    train_manager.train()

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader)
    print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}'.format(auc_test[0], auc_test[1]))


if __name__ == '__main__':

   
    uni_coe = 0.
    env_coe = 0.
    reg_embedding = 0.0001
    reg_dnn = 7e-6
    for seed in [1688723512, 1688723740, 1688738016]:
        main()
    
    print('两个任务AliCPP')
