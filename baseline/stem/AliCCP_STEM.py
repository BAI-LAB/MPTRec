import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import STEM
from multitaskrec.train import TrainManager


def main():
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load data
    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 10000000)
    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 1000000)
    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 10000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    # load model
    AliCCP_Vocabulary_Size.pop("101")  # 干扰特征
    AliCCP_Vocabulary_Size.pop("301")
    device = torch.device(f"cuda:{args.gpu}")
    model = STEM(
        task_num=args.task_num,
        shared_expert_num=1,
        specific_expert_num=1,
        feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5,
        input_size=80,
        expert_dnn_hidden_unit=[128, 64],
        tower_dnn_hidden_unit=[32, 32],
        reg_embedding=1e-6,
        reg_dnn=1e-6,
    )
    model.to(device)

    # load train manager
    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=["CTR", "CVR", "BSI"],
        lr=1e-4,
        epochs=10,
    )

    # compute cost
    # train_manager.compute_cost()

    # train
    train_manager.train_multi_task(args.task_num)

    # save best weight
    # torch.save(train_manager.best_weight, f"stem_alicpp_{args.seed}.pt")

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, args.task_num)
    print(
        "AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}, AUC-Test-BSI:{:.4f}".format(
            auc_test[0], auc_test[1], auc_test[2]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    # 1688723512, 1688723740, 1688738016
    parser.add_argument("--seed", type=int, default=1688723512)
    parser.add_argument("--task_num", type=int, default=3)
    args = parser.parse_args()

    main()
