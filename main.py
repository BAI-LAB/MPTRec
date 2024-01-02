import torch
import argparse
import numpy as np
from utils.functions import load_dataset, load_model, load_train_manager


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'CensusIncome':
        task_name = ['Income', 'Marital']
    elif args.dataset == 'AliCCP':
        task_name = ['CTR', 'CVR']
    else:
        task_name = ['Finish', 'Like']

    # Loading the dataset
    train_loader, val_loader, test_loader = load_dataset(args.dataset, args.seed)
 
    # Loading the model
    model = load_model(args.model, args.dataset)
    device = torch.device("cuda:3")
    model.to(device)

    # Load the TrainManager
    config = {
        'model':model,
        'train_loader':train_loader,
        'val_loader':val_loader
    }
    train_manager = load_train_manager(args.model, args.dataset, config)

    train_manager.compute_cost()
    train_manager.train(args.task_num, args.task_id)

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation(test_loader, args.task_num, args.task_id)
    if args.task_num == 1:
        print('AUC-Test-{}:{:.4f}'.format(task_name[args.task_id], auc_test[0]))
    else:
        print('AUC-Test-{}:{:.4f}, AUC-Test-{}:{:.4f}'.format(task_name[0], auc_test[0], task_name[1], auc_test[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='PLE')
    parser.add_argument('--dataset', type=str, default='CensusIncome')
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1685480945)
    args = parser.parse_args()

    main(args)
