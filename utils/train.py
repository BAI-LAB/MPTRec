import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class TrainManager:
    def __init__(self, model, train_loader, val_loader, task_name, lr, epochs=30, patience=5):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.task_name = task_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.best_weight = None

    def _train_a_batch(self, y, features):
        pred = self.model(features)

        all_loss = self.model.get_l2_reg()
        for task_id in range(len(y)):
            all_loss += self.loss_func(pred[task_id].cpu(), y[task_id].float())

        self.optimizer.zero_grad()
        all_loss.backward()
        self.optimizer.step()

    def train_one_task(self, task_id):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tepoch = tqdm(self.train_loader, unit="batch")

            for y_0, y_1, y_2, features in tepoch:
                y = [y_0, y_1, y_2]
                y = [y[task_id]]
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                self._train_a_batch(y, features)

            auc_val = self.evaluation_one_task(self.val_loader, task_id)
            print('Epoch:{}, AUC-Val-{}:{:.4f}'.format(epoch, self.task_name[task_id], auc_val))

            if auc_val > best_auc_score:
                earlystop_count = 0
                best_auc_score = auc_val
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print('EarlyStopping count {}'.format(earlystop_count))
                if earlystop_count == self.patience:
                    print('EarlyStopping at epoch {}'.format(epoch))
                    break

    def train_multi_task(self, task_num):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tepoch = tqdm(self.train_loader, unit="batch")

            for y_0, y_1, y_2, features in tepoch:
                if task_num == 2:
                    y = [y_0, y_1]
                else:
                    y = [y_0, y_1, y_2]
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                self._train_a_batch(y, features)

            # auc_train = self.evaluation_multi_task(self.train_loader, task_num)
            auc_val = self.evaluation_multi_task(self.val_loader, task_num)

            if task_num == 2:
                print('Epoch:{}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}'.format
                      (epoch, self.task_name[0], auc_val[0], self.task_name[1], auc_val[1]))
            else:
                print('AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}'.format
                      (self.task_name[0], auc_val[0], self.task_name[1], auc_val[1], self.task_name[2], auc_val[2]))

            if sum(auc_val) > best_auc_score:
                earlystop_count = 0
                best_auc_score = sum(auc_val)
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print('EarlyStopping count {}'.format(earlystop_count))
                if earlystop_count == self.patience:
                    print('EarlyStopping at epoch {}'.format(epoch))
                    break

    @torch.no_grad()
    def evaluation_one_task(self, data_loader, task_id):
        self.model.eval()
        device = next(self.model.parameters()).device
        y_true, y_hat = [], []
        for y_0, y_1, y_2, features in data_loader:
            y = [y_0, y_1, y_2]
            for key in features.keys():
                features[key] = features[key].to(device)
            pred = self.model(features)
            y_true.append(y[task_id])
            y_hat.append(pred[0])
        y_true = torch.cat(y_true)
        y_hat = torch.cat(y_hat)
        auc_score = roc_auc_score(y_true.int(), y_hat.cpu())
        return auc_score

    @torch.no_grad()
    def evaluation_multi_task(self, data_loader, task_num):
        self.model.eval()
        device = next(self.model.parameters()).device
        y_true = [[] for _ in range(task_num)]
        y_hat = [[] for _ in range(task_num)]

        for y_0, y_1, y_2, features in data_loader:
            y = [y_0, y_1, y_2]
            for key in features.keys():
                features[key] = features[key].to(device)
            pred = self.model(features)
            for task_id in range(task_num):
                y_true[task_id].append(y[task_id])
                y_hat[task_id].append(pred[task_id])

        auc_score = []
        for task_id in range(task_num):
            y = torch.cat(y_true[task_id])
            pred = torch.cat(y_hat[task_id])
            auc_score.append(roc_auc_score(y.int(), pred.cpu()))

        return auc_score


class SparseSharingTrainManager(TrainManager):
    def __init__(self, model, train_loader, val_loader, all_mask, task_name, lr, epochs=30, patience=5):
        super().__init__(model, train_loader, val_loader, task_name, lr, epochs, patience)
        self.all_mask = all_mask

    def _train_a_batch(self, y, features):
        for task_id in range(len(y)):
            cur_mask = self.all_mask[task_id]
            weights = copy.deepcopy(self.model.shared_bottom.state_dict())

            for name, param in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * cur_mask[name]).to(self.device)

            pred = self.model(features, task_id)
            loss_r = self.loss_func(pred.cpu(), y[task_id].float()) + self.model.get_l2_reg()
            self.optimizer.zero_grad()
            loss_r.backward()

            for name, p in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    grad_tensor = p.grad.data.cpu().numpy()
                    p.grad.data = torch.from_numpy(grad_tensor * cur_mask[name]).to(self.device)
            self.model.shared_bottom.load_state_dict(weights)
            self.optimizer.step()

    @torch.no_grad()
    def evaluation_multi_task(self, data_loader, task_num):
        self.model.eval()
        device = next(self.model.parameters()).device
        y_true = [[] for _ in range(task_num)]
        y_hat = [[] for _ in range(task_num)]

        for task_id in range(task_num):
            weights = copy.deepcopy(self.model.shared_bottom.state_dict())
            cur_mask = self.all_mask[task_id]

            for name, param in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * cur_mask[name]).to(device)

            for y_0, y_1, y_2, features in data_loader:
                if task_num == 2:
                    y = [y_0, y_1]
                else:
                    y = [y_0, y_1, y_2]
                for key in features.keys():
                    features[key] = features[key].to(device)
                pred = self.model(features, task_id)
                y_true[task_id].append(y[task_id])
                y_hat[task_id].append(pred)

            self.model.shared_bottom.load_state_dict(weights)

        auc_score = []
        for task_id in range(task_num):
            y = torch.cat(y_true[task_id])
            pred = torch.cat(y_hat[task_id])
            auc_score.append(roc_auc_score(y.int(), pred.cpu()))

        return auc_score


class CsRecTrainManager(SparseSharingTrainManager):
    def __init__(self, model, train_loader, val_loader, all_mask, task_name, lr, epochs=30, patience=5):
        super().__init__(model, train_loader, val_loader, all_mask, task_name, lr, epochs, patience)
        shared_mask = {}
        for name in self.all_mask[0]:
            if len(self.all_mask) == 2:
                shared_mask[name] = self.all_mask[0][name] * self.all_mask[1][name]
            elif len(self.all_mask) == 3:
                shared_mask[name] = self.all_mask[0][name] * self.all_mask[1][name] * self.all_mask[2][name]
        self.contrastive_mask = {}
        for name, mask in self.all_mask[0].items():
            p = np.random.random(mask.shape)
            self.contrastive_mask[name] = np.where(p > 1, 0, 1 - shared_mask[name])

    def _train_a_batch(self, y, features):
        for task_id in range(len(y)):
            cur_mask = self.all_mask[task_id]
            weights = copy.deepcopy(self.model.shared_bottom.state_dict())

            for name, param in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * self.contrastive_mask[name]).to(self.device)

            pred = self.model(features, task_id)
            loss_r_hat = - self.loss_func(pred.cpu(), y[task_id].float())
            self.optimizer.zero_grad()
            loss_r_hat.backward()

            grads = {}
            for name, p in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    tensor = p.grad.data.cpu().numpy()
                    grads[name] = torch.from_numpy(tensor * self.contrastive_mask[name]).to(self.device)

            self.model.shared_bottom.load_state_dict(weights)

            for name, param in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * cur_mask[name]).to(self.device)

            pred = self.model(features, task_id)
            loss_r = self.loss_func(pred.cpu(), y[task_id].float()) + self.model.get_l2_reg()
            self.optimizer.zero_grad()
            loss_r.backward()

            for name, p in self.model.shared_bottom.named_parameters():
                if 'weight' in name:
                    grad_tensor = p.grad.data.cpu().numpy()
                    p.grad.data = torch.from_numpy(grad_tensor * cur_mask[name]).to(self.device) + grads[name]
            self.model.shared_bottom.load_state_dict(weights)
            self.optimizer.step()


class MPTRecTrainManager(TrainManager):
    def __init__(self, model, train_loader, val_loader, env_ids, task_name, lr, batch_size, uni_coe, env_coe, epochs=30, patience=5):
        super().__init__(model, train_loader, val_loader, task_name, lr, epochs, patience)
        self.env_ids = env_ids
        self.env_loss_func = nn.NLLLoss()
        self.batch_size = batch_size
        self.uni_coe = uni_coe
        self.env_coe = env_coe
        
        self.uni_loss_0_list = []
        self.uni_loss_1_list = []
        self.fused_loss_0_list = []
        self.fused_loss_1_list = []
        self.env_loss_list = []

    def train_two_task(self):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            uni_loss_0_sum = 0
            uni_loss_1_sum = 0
            fused_loss_0_sum = 0
            fused_loss_1_sum = 0
            env_loss_sum = 0

            tepoch = tqdm(self.train_loader, unit="batch")
            for step, (y_0, y_1, _, features) in enumerate(tepoch):
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                p = float(step + (epoch - 1) * self.batch_size) / float(
                    self.epochs * self.batch_size)
                alpha = 2. / (1. + np.exp(-10. * p)) - 1.

                output = self.model(features, alpha)

                batch_env_ids = self.env_ids[self.batch_size * step:self.batch_size * (step + 1)]
                uni_loss_0 = self.loss_func(output['uni_preds'][0].cpu(), y_0.float())
                uni_loss_1 = self.loss_func(output['uni_preds'][1].cpu(), y_1.float())
                fused_loss_0 = self.loss_func(output['fused_preds'][0].cpu(), y_0.float())
                fused_loss_1 = self.loss_func(output['fused_preds'][1].cpu(), y_1.float())
                env_loss = self.env_loss_func(output['env_pred'].cpu(), batch_env_ids)
                # loss = fused_loss_0 + fused_loss_1 + self.uni_coe * (uni_loss_0 + uni_loss_1) + \
                #        self.env_coe * env_loss + self.model.get_l2_reg()
                loss = fused_loss_0 + fused_loss_1 + self.model.get_l2_reg()

                uni_loss_0_sum += uni_loss_0
                uni_loss_1_sum += uni_loss_1
                fused_loss_0_sum += fused_loss_0
                fused_loss_1_sum += fused_loss_1
                env_loss_sum += env_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            uni_loss_0_sum /= len(self.train_loader)
            uni_loss_1_sum /= len(self.train_loader)
            fused_loss_0_sum /= len(self.train_loader)
            fused_loss_1_sum /= len(self.train_loader)
            env_loss_sum /= len(self.train_loader)
            
            print('uni_loss_0:{:.4f}, uni_loss_1:{:.4f}, fuse_loss_0:{:.4f}, fuse_loss_1:{:.4f}, env_loss:{:.4f}'.
                  format(uni_loss_0_sum, uni_loss_1_sum, fused_loss_0_sum, fused_loss_1_sum, env_loss_sum))
            
            self.uni_loss_0_list.append(uni_loss_0_sum.item())
            self.uni_loss_1_list.append(uni_loss_1_sum.item())
            self.fused_loss_0_list.append(fused_loss_0_sum.item())
            self.fused_loss_1_list.append(fused_loss_1_sum.item())
            self.env_loss_list.append(env_loss_sum.item())

            if epoch % 2 == 0:
                self.env_ids = self.cluster_2()

            # auc_train = self.evaluation_two_task(self.train_loader)
            auc_val = self.evaluation_two_task(self.val_loader)
            print('Epoch:{}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}, '.format(epoch, self.task_name[0], auc_val[0], 
                                                                            self.task_name[1], auc_val[1]))

            if sum(auc_val) > best_auc_score:
                earlystop_count = 0
                best_auc_score = sum(auc_val)
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print('EarlyStopping count {}'.format(earlystop_count))
                if earlystop_count == self.patience:
                    print('EarlyStopping at epoch {}'.format(epoch))
                    break

    def train_three_task(self):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            tepoch = tqdm(self.train_loader, unit="batch")
            for step, (y_0, y_1, y_2, features) in enumerate(tepoch):
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                p = float(step + (epoch - 1) * self.batch_size) / float(
                    self.epochs * self.batch_size)
                alpha = 2. / (1. + np.exp(-10. * p)) - 1.

                output = self.model(features, alpha)

                batch_env_ids = self.env_ids[self.batch_size * step:self.batch_size * (step + 1)]
                uni_loss_0 = self.loss_func(output['uni_preds'][0].cpu(), y_0.float()) 
                uni_loss_1 = self.loss_func(output['uni_preds'][1].cpu(), y_1.float()) 
                uni_loss_2 = self.loss_func(output['uni_preds'][2].cpu(), y_2.float())
                fused_loss_0 = self.loss_func(output['fused_preds'][0].cpu(), y_0.float()) 
                fused_loss_1 = self.loss_func(output['fused_preds'][1].cpu(), y_1.float()) 
                fused_loss_2 = self.loss_func(output['fused_preds'][2].cpu(), y_2.float()) 
                env_loss = self.env_loss_func(output['env_pred'].cpu(), batch_env_ids)
                loss = fused_loss_0 + fused_loss_1 + fused_loss_2 + self.uni_coe * (uni_loss_0 + uni_loss_1 + uni_loss_2) + \
                       self.env_coe * env_loss + self.model.get_l2_reg()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 2 == 0:
                self.env_ids = self.cluster_3()

            auc_val = self.evaluation_three_task(self.val_loader)
            print('Epoch:{}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}'
                  .format(epoch, self.task_name[0], auc_val[0], self.task_name[1], auc_val[1], self.task_name[2], auc_val[2]))

            if sum(auc_val) > best_auc_score:
                earlystop_count = 0
                best_auc_score = sum(auc_val)
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print('EarlyStopping count {}'.format(earlystop_count))
                if earlystop_count == self.patience:
                    print('EarlyStopping at epoch {}'.format(epoch))
                    break

    @torch.no_grad()
    def cluster_2(self):
        self.model.eval()
        loss_func = nn.BCELoss(reduction='none')
        new_env_tensors_list = []
        for y_0, y_1, _, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.cluster_predict(features)
            loss_0 = loss_func(pred[0].cpu(), y_0.float())
            loss_1 = loss_func(pred[1].cpu(), y_1.float())
            loss = torch.stack([loss_0, loss_1], dim=1)
            new_env_tensors_list.append(torch.argmin(loss, dim=1))
        all_new_env_tensors = torch.cat(new_env_tensors_list, dim=0)
        env_diff = self.env_ids - all_new_env_tensors
        diff_num = torch.nonzero(env_diff != 0).shape[0]
        counts = torch.unique(all_new_env_tensors, return_counts=True)
        envs = [0, 0]
        for i in range(len(counts[0])):
            envs[counts[0][i]] = counts[1][i]
        print('diff_num:{}, env_0:{}, env_1:{}'.format(diff_num, envs[0], envs[1]))
        return all_new_env_tensors
    
    @torch.no_grad()
    def cluster_3(self):
        self.model.eval()
        loss_func = nn.BCELoss(reduction='none')
        new_env_tensors_list = []
        for y_0, y_1, y_2, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.cluster_predict(features)
            loss_0 = loss_func(pred[0].cpu(), y_0.float())
            loss_1 = loss_func(pred[1].cpu(), y_1.float())
            loss_2 = loss_func(pred[2].cpu(), y_2.float())
            loss = torch.stack([loss_0, loss_1, loss_2], dim=1)
            new_env_tensors_list.append(torch.argmin(loss, dim=1))
        all_new_env_tensors = torch.cat(new_env_tensors_list, dim=0)
        env_diff = self.env_ids - all_new_env_tensors
        diff_num = torch.nonzero(env_diff != 0).shape[0]
        counts = torch.unique(all_new_env_tensors, return_counts=True)
        envs = [0, 0, 0]
        for i in range(len(counts[0])):
            envs[counts[0][i]] = counts[1][i]
        print('diff_num:{}, env_0:{}, env_1:{}, env_2:{}'.format(diff_num, envs[0], envs[1], envs[2]))
        return all_new_env_tensors

    @torch.no_grad()
    def evaluation_two_task(self, data_loader):
        self.model.eval()
        y0_true, y1_true, y0_hat, y1_hat = [], [], [], []
        for y_0, y_1, _, features in data_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.predict(features)
            y0_true.append(y_0)
            y1_true.append(y_1)
            y0_hat.append(pred[0])
            y1_hat.append(pred[1])
        y0_true = torch.cat(y0_true)
        y1_true = torch.cat(y1_true)
        y0_hat = torch.cat(y0_hat)
        y1_hat = torch.cat(y1_hat)
        auc_score_0 = roc_auc_score(y0_true.int(), y0_hat.cpu())
        auc_score_1 = roc_auc_score(y1_true.int(), y1_hat.cpu())
        return [auc_score_0, auc_score_1]

    @torch.no_grad()
    def evaluation_three_task(self, data_loader):
        self.model.eval()
        y0_true, y1_true, y2_true = [], [], []
        y0_hat, y1_hat, y2_hat = [], [], []
        for y_0, y_1, y_2, features in data_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.predict(features)
            y0_true.append(y_0)
            y1_true.append(y_1)
            y2_true.append(y_2)
            y0_hat.append(pred[0])
            y1_hat.append(pred[1])
            y2_hat.append(pred[2])
        y0_true = torch.cat(y0_true)
        y1_true = torch.cat(y1_true)
        y2_true = torch.cat(y2_true)
        y0_hat = torch.cat(y0_hat)
        y1_hat = torch.cat(y1_hat)
        y2_hat = torch.cat(y2_hat)
        auc_score_0 = roc_auc_score(y0_true.int(), y0_hat.cpu())
        auc_score_1 = roc_auc_score(y1_true.int(), y1_hat.cpu())
        auc_score_2 = roc_auc_score(y2_true.int(), y2_hat.cpu())
        return [auc_score_0, auc_score_1, auc_score_2]
