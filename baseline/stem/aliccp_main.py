import argparse
import os


import numpy as np

import torch

import wandb

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader


from config import AliCCP_Vocabulary_Size

from multitaskrec.dataset import AliCCPDataset

from multitaskrec.model import STEM

from multitaskrec.train import MultiTaskTrainManager


os.environ["WANDB_MODE"] = "dryrun"



def main(seed, gpu):

    # set random seed

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    # load dataset

    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 10000000)

    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 1000000)

    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 10000000)

    train_loader = DataLoader(train_dataset, batch_size=2000)

    val_loader = DataLoader(val_dataset, batch_size=2000)

    test_loader = DataLoader(test_dataset, batch_size=2000)


    # build model

    model = STEM(

        task_num=2,

        shared_expert_num=1,

        specific_expert_num=1,

        feature_vocabulary=AliCCP_Vocabulary_Size,

        embedding_size=5,

        input_size=90,

        expert_dnn_hidden_unit=[128, 64],

        tower_dnn_hidden_unit=[32, 32],

        reg_embedding=1e-6,

        reg_dnn=1e-6,

        dropout=[0.1, 0.3],
    )

    device = torch.device(f"cuda:{gpu}")

    model.to(device)

    train_manager = MultiTaskTrainManager(
        model=model,
        train_loader=train_loader,

        val_loader=val_loader,

        task_name=["CTR", "CVR"],

        lr=1e-4,

        epochs=30,

        patience=5,
    )


    # counting parameters and floating-point operands
    train_manager.compute_cost()


    # training

    wandb.init(

        project="multitaskrec",

        config={"model": "stem", "dataset": "AliCCP", "seed": seed},
    )
    train_manager.train()

    wandb.finish()


    # evaluation

    model.load_state_dict(train_manager.best_weight)

    torch.save(model.state_dict(), "mmoe_alicpp.pt")

    auc_test = train_manager.evaluation(test_loader)

    print("AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}".format(auc_test[0], auc_test[1]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    # 1688723512, 1688723740, 1688738016, 1688749593, 1688762746

    parser.add_argument("--seed", type=int, default=1688723512)

    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args.seed, args.gpu)

