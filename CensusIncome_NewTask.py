import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager


@torch.no_grad()
def evaluation(newtask, invchar, data_loader):
    newtask.eval()
    device = next(newtask.parameters()).device
    y_true, y_hat = [], []
    for _, _, y, features in data_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = invchar.get_infos(features)
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        y_true.append(y)
        y_hat.append(pred)
    y_true = torch.cat(y_true)
    y_hat = torch.cat(y_hat)
    auc_score = roc_auc_score(y_true.int(), y_hat.cpu())
    return auc_score


def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    CensusIncome_Vocabulary_Size.pop("education")
    device = torch.device(f"cuda:{args.gpu}")
    mptrec = MPTRec(
        num_tasks=2,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=123,
        expert_dnn_hidden_units=[256, 128],
        tower_dnn_hidden_units=[64, 32],
        reg_embedding=args.reg_embedding,
        reg_dnn=args.reg_dnn,
        device=device,
    )
    mptrec.to(device)

    newtask = NewTask(
        input_size=123,
        rep_dim=128,
        tower_dnn_hidden_units=[64, 32],
        reg_dnn=args.reg_dnn,
        device=device,
    )
    newtask.to(device)

    print("-" * 32, "Multi-task pre-training phase", "-" * 32)
    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=["Income", "Marital"],
        lr=1e-3,
        batch_size=256,
        uni_coe=args.uni_coe,
        env_coe=args.env_coe,
        epochs=10,
    )
    train_manager.train_two_task()
    mptrec.load_state_dict(train_manager.best_weight)

    print("-" * 32, "New task generalization phase", "-" * 32)
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
        print("AUC-Val-Education:{:.4f}".format(auc_val))
        if auc_val > best_auc_score:
            earlystop_count = 0
            best_auc_score = auc_val
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            print("EarlyStopping count {}".format(earlystop_count))
            if earlystop_count == patience:
                print("EarlyStopping at epoch {}".format(epoch))
                break

    newtask.load_state_dict(best_weight)
    auc_test = evaluation(newtask, mptrec, test_loader)
    print("AUC-Test-Education:{:.4f}".format(auc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uni_coe", type=float, default=0)
    parser.add_argument("--env_coe", type=float, default=0)
    parser.add_argument("--reg_embedding", type=float, default=0.006)
    parser.add_argument("--reg_dnn", type=float, default=3e-5)
    parser.add_argument("--gpu", type=int, default=0)
    # 1685480945, 1685463909, 1685477428
    parser.add_argument("--seed", type=int, default=1685480945)

    args = parser.parse_args()
    main(args)
