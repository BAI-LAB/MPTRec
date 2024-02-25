import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import SparseSharing
from multitaskrec.train import SparseSharingTrainManager


def main():
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load dataset
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=args.seed
    )
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # build model
    model = SparseSharing(
        num_tasks=2,
        feature_vocabulary=CensusIncome_Vocabulary_Size,
        embedding_size=4,
        input_size=127,
        shared_dnn_hidden_units=[256, 128],
        tower_dnn_hidden_units=[64, 32],
        reg_embedding=3e-4,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model.to(device)

    # build train manager
    train_manager = SparseSharingTrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mask_path=f"mask/CensusIncome/mask_{seed}.pt",
        task_name=["Income", "Marital"],
        lr=1e-3,
        epochs=30,
        patience=5,
    )

    # counting parameters and floating-point operands
    train_manager.compute_cost()

    # training
    train_manager.train()

    # testing
    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader)
    print(
        "AUC-Test-Income:{:.4f}, AUC-Test-Marital:{:.4f}".format(
            auc_test[0], auc_test[1]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1685480945)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    main(args)
