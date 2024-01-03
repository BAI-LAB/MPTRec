import sys
import copy
import torch
import optuna
import warnings
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append('/data/hl/MultiTask/')

from utils.dataset import AliCppDataset
from utils.models import MPTRec, NewTask
from utils.train import MPTRecTrainManager
from utils.config import AliCpp_Vocabulary_Size

warnings.filterwarnings('ignore')


@torch.no_grad()
def evaluation(newtask, invchar, data_loader):
    newtask.eval()
    device = next(newtask.parameters()).device
    y_true, y_hat = [], []
    for _, _, y, features in data_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, invariant_rep, variant_reps, env_embeddings = invchar.get_infos(features)
        pred = newtask(dnn_input, invariant_rep, variant_reps, env_embeddings)
        y_true.append(y)
        y_hat.append(pred)
    y_true = torch.cat(y_true)
    y_hat = torch.cat(y_hat)
    auc_score = roc_auc_score(y_true.int(), y_hat.cpu())
    return auc_score


def main():
    train_dataset = AliCppDataset('/data/hl/MultiTask/data/Ali-CCP/ctr_cvr.train', 2000000)
    val_dataset = AliCppDataset('/data/hl/MultiTask/data/Ali-CCP/ctr_cvr.dev', 200000)
    test_dataset = AliCppDataset('/data/hl/MultiTask/data/Ali-CCP/ctr_cvr.test', 2000000)
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

        # uni_coe = 0.5
        # env_coe = 0.8
        reg_embedding = 0.0001
        reg_dnn = 7e-6
       
        device = torch.device("cuda:5")
        mptrec = MPTRec(
            num_tasks=2,
            feature_vocabulary=AliCpp_Vocabulary_Size,
            embedding_size=5,
            input_size=80,
            expert_dnn_hidden_units=(128, 64),
            tower_dnn_hidden_units=(32, 32),
            dropout=(0.1, 0.3),
            reg_embedding=reg_embedding,
            reg_dnn=reg_dnn,
            device=device
        )
        mptrec.to(device)

        newtask = NewTask(
            input_size=80,
            rep_dim=64,
            tower_dnn_hidden_units=(32, 32),
            reg_dnn=reg_dnn,
            device=device
        )
        newtask.to(device)

        print('-' * 32, 'Multi-task pre-training phase', '-' * 32)
        train_manager = MPTRecTrainManager(
            model=mptrec,
            train_loader=train_loader,
            val_loader=val_loader,
            env_ids=env_ids,
            task_name=['CTR', 'CVR'],
            lr=1e-4,
            batch_size=2000,
            uni_coe=uni_coe,
            env_coe=env_coe
        )
        train_manager.train_two_task()
        mptrec.load_state_dict(train_manager.best_weight)

        print('-' * 32, 'New task generalization phase', '-' * 32)
        optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-4)
        loss_func = nn.BCELoss()
        epochs = 30
        patience = 5
        earlystop_count = 0
        best_auc_score = 0
        best_weight = None

        for epoch in range(1, epochs + 1):
            newtask.train()
            for _, _, y, features in train_loader:
                for key in features.keys():
                    features[key] = features[key].to(device)
                dnn_input, invariant_rep, variant_reps, env_embeddings = mptrec.get_infos(features)
                pred = newtask(dnn_input, invariant_rep, variant_reps, env_embeddings)
                loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            auc_val = evaluation(newtask, mptrec, val_loader)
            if auc_val > best_auc_score:
                earlystop_count = 0
                best_auc_score = auc_val
                best_weight = copy.deepcopy(newtask.state_dict())
            else:
                earlystop_count += 1
                print('EarlyStopping count {}'.format(earlystop_count))
                if earlystop_count == patience:
                    print('EarlyStopping at epoch {}'.format(epoch))
                    break

        newtask.load_state_dict(best_weight)
        auc_test = evaluation(newtask, mptrec, test_loader)
        print('AUC-Test-BSI:{:.4f}'.format(auc_test))
    
        return auc_test
    
    study = optuna.create_study(study_name='test', direction='maximize')
    study.optimize(objective, n_trials=30)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)


if __name__ == '__main__':
    seed = 1688723512
    main()
   