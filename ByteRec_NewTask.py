import copy
import torch
import warnings
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from utils.models import MPTRec, NewTask
from utils.dataset import ByteRecDataset
from utils.train import MPTRecTrainManager
from utils.config import ByteRec_Vocabulary_Size

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

    uni_coe = 0.9
    env_coe = 0.1
    reg_embedding = 0.0001
    reg_dnn = 7e-6

    train_dataset = ByteRecDataset('/home/hl/MultiTask/data/ByteRec/train.gz')
    val_dataset = ByteRecDataset('/home/hl/MultiTask/data/ByteRec/val.gz')
    test_dataset = ByteRecDataset('/home/hl/MultiTask/data/ByteRec/test.gz')
    train_loader = DataLoader(train_dataset, batch_size=4000)
    val_loader = DataLoader(val_dataset, batch_size=4000)
    test_loader = DataLoader(test_dataset, batch_size=4000)
    env_ids = torch.load('/home/hl/MultiTask/data/ByteRec/env_id.gz')

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

    newtask = NewTask(
        input_size=32,
        rep_dim=64,
        tower_dnn_hidden_units=(32, 32),
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
    torch.save(train_manager.best_weight, f'/home/hl/MultiTask/ByteRec_{seed}.pt')
    print('AUC-Test-Finish:{:.4f}, AUC-Test-Like:{:.4f}'.format(auc_test[0], auc_test[1]))

    print('-' * 32, 'New task generalization phase', '-' * 32)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-4)
    loss_func = nn.BCELoss()
    epochs = 10
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
    print('AUC-Test-Duration_time:{:.4f}'.format(auc_test))


if __name__ == '__main__':
    for seed in [1688723512]:
        main()
