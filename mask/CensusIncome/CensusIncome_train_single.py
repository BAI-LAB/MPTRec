import sys


import copy


import torch


import warnings


import numpy as np


from tqdm import tqdm


from torch import nn


from torch.utils.data import DataLoader


from sklearn.metrics import roc_auc_score


from sklearn.model_selection import train_test_split



sys.path.append('/data/hl/MultiTask/')



from multitaskrec.train import MultiTaskTrainManager


from multitaskrec.model import SharedBottom


from multitaskrec.functions import count_prune_rate


from multitaskrec.dataset import CensusIncomeDataset


from config import CensusIncome_Vocabulary_Size



warnings.filterwarnings('ignore')




def train_single():


    torch.manual_seed(seed)


    torch.cuda.manual_seed(seed)


    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



    train_dataset = CensusIncomeDataset('/data/hl/MultiTask/data/CensusIncome/train.gz')


    test_dataset = CensusIncomeDataset('/data/hl/MultiTask/data/CensusIncome/test.gz')


    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)


    train_loader = DataLoader(train_dataset, batch_size=256)


    val_loader = DataLoader(val_dataset, batch_size=256)



    model = SharedBottom(


        num_tasks=2,


        feature_vocabulary=CensusIncome_Vocabulary_Size,


        embedding_size=4,


        input_size=127,


        shared_dnn_hidden_units=(256, 128),


        tower_dnn_hidden_units=(64, 32),


        reg_embedding=3e-4,


        reg_dnn=0
    )


    device = torch.device("cuda:5")


    model.to(device)



    print('Start warm up!!!')


    train_manager = MultiTaskTrainManager(


        model=model,


        train_loader=train_loader,


        val_loader=val_loader,


        task_name=['Income', 'Marital'],


        lr=1e-3,


        epochs=5
    )


    train_manager.train(2)


    print('End warm up!!!')
    


    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)


    loss_func = nn.BCELoss()


    task_name = ['Income', 'Marital']


    all_mask = []
    


    for task_id in range(2):


        model.load_state_dict(train_manager.best_weight)


        cur_mask = make_mask(model)


        epochs = 5


        patience = 5


        iteration = 10


        prune_percent = 10


        best_prune = 0


        prune_rate = 0
        


        for _ite in range(0, iteration):


            if not _ite == 0:


                # Prune


                for name, param in model.shared_network.named_parameters():


                    if 'weight' in name:


                        tensor = param.data.cpu().numpy()


                        tensor = tensor * cur_mask[name]


                        alive = tensor[np.nonzero(tensor)] 


                        percentile_value = np.percentile(abs(alive), prune_percent)


                        new_mask = np.where(abs(tensor) < percentile_value, 0, cur_mask[name])


                        cur_mask[name] = new_mask



                # Initial


                model.load_state_dict(train_manager.best_weight)


                for name, param in model.shared_network.named_parameters():


                    if 'weight' in name:


                        tensor = param.data.cpu().numpy()


                        param.data = torch.from_numpy(tensor * cur_mask[name]).to(device)



                prune_rate = count_prune_rate(cur_mask)


                print(f"\n--- Pruning Level [{task_id}:{_ite}/{iteration - 1}]: ---")


                print('prune_rate:{}'.format(prune_rate))



            best_auc_score = 0


            earlystop_count = 0


            best_weight = None


            for epoch in range(0, epochs):
                model.train()


                tepoch = tqdm(train_loader, unit="batch")


                for y_0, y_1, features in tepoch:


                    y = [y_0, y_1]


                    for key in features.keys():


                        features[key] = features[key].to(device)



                    pred = model(features)


                    loss_r = loss_func(pred[task_id].cpu(), y[task_id].float()) + model.get_l2_reg()


                    optimizer.zero_grad()


                    loss_r.backward()


                    for name, p in model.shared_network.named_parameters():


                        if 'weight' in name:


                            grad_tensor = p.grad.data.cpu().numpy()


                            p.grad.data = torch.from_numpy(grad_tensor * cur_mask[name]).to(device)


                    optimizer.step()



                auc_val = evaluation(model, val_loader, task_id)


                print('AUC-Val-{}:{:.4f}'.format(task_name[task_id], auc_val))



                if auc_val > best_auc_score:


                    earlystop_count = 0


                    best_auc_score = auc_val


                    best_weight = copy.deepcopy(model.state_dict())


                else:


                    earlystop_count += 1


                    print('EarlyStopping count {}'.format(earlystop_count))


                    if earlystop_count == patience:


                        print('EarlyStopping at epoch {}'.format(epoch))


                        break



            if best_weight:


                model.load_state_dict(best_weight)



            if prune_rate > 0.4 and best_auc_score > best_prune:


                best_prune = best_auc_score


                print('prune_time:{}'.format(_ite))


                best_mask = cur_mask


        all_mask.append(best_mask)


    torch.save(all_mask, f'/data/hl/MultiTask/mask/CensusIncome/mask_{seed}.pt')




@torch.no_grad()


def evaluation(model, data_loader, task_id):


    model.eval()


    device = next(model.parameters()).device


    y_true, y_hat = [], []


    for y_0, y_1, features in data_loader:


        y = [y_0, y_1]


        for key in features.keys():


            features[key] = features[key].to(device)


        pred = model(features)


        y_true.append(y[task_id])


        y_hat.append(pred[task_id])


    y = torch.cat(y_true)


    pred = torch.cat(y_hat)


    auc_score = roc_auc_score(y.int(), pred.cpu())
    return auc_score




def make_mask(model):


    mask = {}


    for name, param in model.shared_network.named_parameters():


        if 'weight' in name:


            tensor = param.data.cpu().numpy()


            mask[name] = np.ones_like(tensor)


    return mask




if __name__ == '__main__':


    for seed in [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]:


        train_single()


