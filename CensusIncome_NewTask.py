import copy
import torch
import warnings
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
 
    train_dataset = CensusIncomeDataset('/home/huangle/MultiTask/dataset/CensusIncome/#train.gz')
    test_dataset = CensusIncomeDataset('/home/huangle/MultiTask/dataset/CensusIncome/#test.gz')
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    env_ids = torch.load('/home/huangle/MultiTask/dataset/CensusIncome/#env_id.gz')

    device = torch.device("cuda:0")
    mptrec = MPTRec(
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
    mptrec.to(device)
    mptrec.base_network.load_state_dict(torch.load('/home/huangle/MultiTask/ci_base.pt'))
    mptrec.embedding_networks.load_state_dict(torch.load('/home/huangle/MultiTask/ci_embedding.pt'))

    newtask = NewTask(
        input_size=123,
        rep_dim=128,
        tower_dnn_hidden_units=(64, 32),
        reg_dnn=reg_dnn,
        device=device
    )
    newtask.to(device)

    # from utils.functions import compute_cost_2
    # compute_cost_2(mptrec, newtask, train_loader)

    print('-' * 32, 'Multi-task pre-training phase', '-' * 32)
    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=['income', 'marital'],
        lr=1e-3,
        batch_size=256,
        uni_coe=uni_coe,
        env_coe=env_coe
    )
    train_manager.train_two_task()

    mptrec.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_two_task(test_loader)
    print('AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}'.format(auc_test[0], auc_test[1]))

    print('-' * 32, 'New task generalization phase', '-' * 32)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-3)
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
            dnn_input, uni_rep, prop_reps, env_embeddings = mptrec.get_infos(features)
            pred = newtask(dnn_input, uni_rep, prop_reps, env_embeddings)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        auc_val = evaluation(newtask, mptrec, val_loader)
        print('AUC-Val-Education:{:.4f}'.format(auc_val))
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
    print('AUC-Test-Education:{:.4f}'.format(auc_test))


if __name__ == '__main__':
    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.006
    reg_dnn = 3e-5
    # for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:
    #     main()

    seed = 1685480945
    main()
