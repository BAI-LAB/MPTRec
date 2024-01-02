import sys
import time
import torch
import optuna
import warnings
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('/home/hl/MultiTask/')

from utils.models import MPTRec
from utils.dataset import AliCppDataset
from utils.train import MPTRecTrainManager
from utils.config import AliCpp_Vocabulary_Size

warnings.filterwarnings('ignore')


def main():
    train_dataset = AliCppDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.train', 2000000)
    val_dataset = AliCppDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.dev', 200000)
    test_dataset = AliCppDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.test', 2000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))

    def objective(trail):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        uni_coe = trail.suggest_float('uni_coe', 0, 1, step=1e-1)
        env_coe = trail.suggest_float('env_coe', 0, 1, step=1e-1)
       
        # uni_coe = 0.3
        # env_coe = 0.9
        reg_embedding = 1e-4
        reg_dnn = 7e-6

        device = torch.device("cuda:5")
        model = MPTRec(
            num_tasks=2,
            feature_vocabulary=AliCpp_Vocabulary_Size,
            embedding_size=5,
            input_size=90,
            expert_dnn_hidden_units=(128, 64),
            tower_dnn_hidden_units=(32, 32),
            reg_embedding=reg_embedding,
            reg_dnn=reg_dnn,
            dropout=(0.1, 0.3),
            device=device
        )
        model.to(device)

        train_manager = MPTRecTrainManager(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            env_ids=env_ids,
            task_name=['CTR', 'CVR'],
            lr=1e-4,
            batch_size=2000,
            uni_coe=uni_coe,
            env_coe=env_coe,
            epochs=20
        )
        train_manager.train_two_task()

        model.load_state_dict(train_manager.best_weight)
        auc_test = train_manager.evaluation_two_task(test_loader)
        print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}'.format(auc_test[0], auc_test[1]))
        return auc_test[0]

    study = optuna.create_study(study_name='test', direction='maximize')
    study.optimize(objective, n_trials=30)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)


if __name__ == '__main__':
    seed = 1688723512
    main()
