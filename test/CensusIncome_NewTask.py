import sys
import copy
import torch
import optuna
import warnings
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append('/data/hl/MultiTask/')

from utils.models import MPTRec, NewTask
from utils.train import MPTRecTrainManager
from utils.dataset import CensusIncomeDataset
from utils.config import CensusIncome_Vocabulary_Size

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
    def objective(trail):
        train_dataset = CensusIncomeDataset('/data/hl/MultiTask/data/CensusIncome/#train.gz')
        test_dataset = CensusIncomeDataset('/data/hl/MultiTask/data/CensusIncome/#test.gz')
        val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
        train_loader = DataLoader(train_dataset, batch_size=256)
        val_loader = DataLoader(val_dataset, batch_size=256)
        test_loader = DataLoader(test_dataset, batch_size=256)
        env_ids = torch.load('/data/hl/MultiTask/data/CensusIncome/#env_id.gz')

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        uni_coe = trail.suggest_float('uni_coe', 0, 1, step=1e-1)
        env_coe = trail.suggest_float('env_coe', 0, 1, step=1e-1)

        # uni_coe = 0.7
        # env_coe = 0.9
        reg_embedding = 0.006
        reg_dnn = 3e-5

        device = torch.device("cuda:3")
        invchar = MPTRec(
            num_tasks=2,
            feature_vocabulary=CensusIncome_Vocabulary_Size,
            embedding_size=4,
            input_size=123,
            expert_dnn_hidden_units=(256, 128),
            tower_dnn_hidden_units=(64, 32),
            reg_embedding=reg_embedding,
            reg_dnn=reg_dnn,
            device=device
        )
        invchar.to(device)

        newtask = NewTask(
            input_size=123,
            rep_dim=128,
            tower_dnn_hidden_units=(64, 32),
            reg_dnn=reg_dnn,
            device=device
        )
        newtask.to(device)

        print('-' * 32, 'Multi-task pre-training phase', '-' * 32)
        train_manager = MPTRecTrainManager(
            model=invchar,
            train_loader=train_loader,
            val_loader=val_loader,
            env_ids=env_ids,
            task_name=['income', 'marital'],
            lr=1e-3,
            batch_size=256,
            epochs=1,
            uni_coe=uni_coe,
            env_coe=env_coe
        )
        train_manager.train_two_task()
        invchar.load_state_dict(train_manager.best_weight)

        print('-' * 32, 'New task generalization phase', '-' * 32)
        optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-3)
        loss_func = nn.BCELoss()
        epochs = 1
        patience = 5
        earlystop_count = 0
        best_auc_score = 0
        best_weight = None

        for epoch in range(1, epochs + 1):
            newtask.train()
            for _, _, y, features in train_loader:
                for key in features.keys():
                    features[key] = features[key].to(device)
                dnn_input, uni_rep, prop_reps, env_embeddings = invchar.get_infos(features)
                pred = newtask(dnn_input, uni_rep, prop_reps, env_embeddings)
                loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            auc_val = evaluation(newtask, invchar, val_loader)
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
        auc_test = evaluation(newtask, invchar, test_loader)
        print('AUC-Test-Education:{:.4f}'.format(auc_test))

        return auc_test
    
    study = optuna.create_study(study_name='test', direction='maximize')
    study.optimize(objective, n_trials=3)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)


if __name__ == '__main__':
    seed = 1685480945
    main()
