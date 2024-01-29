import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb


class SingleTaskTrainManager:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_id: int,
        task_name: str,
        lr: float,
        epochs: int,
        patience: int,
        wandb_log: bool = False,
    ):
        """Train Manager for SingleTask

        Args:
            model: model
            train_loader: train data loader
            val_loader: val data loader
            task_id: task id
            task_name: task name
            lr: learning rate
            epochs: epochs
            patience: patience
            wandb_log: whether to log to wandb
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_id = task_id
        self.task_name = task_name
        self.epochs = epochs
        self.patience = patience
        self.wandb_log = wandb_log
        self.best_weight = None

    def train(self):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0

            for y_0, y_1, features in tqdm(self.train_loader, unit="batch"):
                y = [y_0, y_1]
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                pred = self.model(features)
                batch_loss = (
                    self.loss_func(pred.cpu(), y[task_id].float())
                    + self.model.get_l2_reg()
                )

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss

            epoch_loss /= len(self.train_loader)
            auc_val = self.evaluation(self.val_loader)
            if self.wandb_log:
                wandb.log(
                    {
                        "Loss": epoch_loss,
                        f"AUC-Val-{self.task_name}": auc_val,
                    }
                )
            print(
                "Epoch:{}, Loss:{}, AUC-Val-{}:{:.4f}".format(
                    epoch,
                    epoch_loss,
                    self.task_name,
                    auc_val,
                )
            )

            if auc_val > best_auc_score:
                earlystop_count = 0
                best_auc_score = auc_val
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print("EarlyStopping count {}".format(earlystop_count))
                if earlystop_count == self.patience:
                    print("EarlyStopping at epoch {}".format(epoch))
                    break

    @torch.no_grad()
    def evaluation(self, data_loader: DataLoader):
        self.model.eval()
        y_true = []
        y_hat = []

        for y_0, y_1, features in data_loader:
            y = [y_0, y_1]
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model(features)
            y_true.append(y[self.task_id])
            y_hat.append(pred)

        y_true = torch.cat(y_true)
        y_hat = torch.cat(y_hat)
        auc_score = roc_auc_score(y_true.int(), y_hat.cpu())

        return auc_score

    def count_params(self):
        trainable_params_num, total_params_num = 0, 0
        for _, params in self.model.named_parameters():
            total_params_num += params.numel()
            if params.requires_grad:
                trainable_params_num += params.numel()
        print("=" * 64)
        print("Total params: {}".format(total_params_num))
        print("Trainable params: {}".format(trainable_params_num))
        print("=" * 64)

    def compute_cost(self):
        self.count_params()
        for _, _, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            flops = FlopCountAnalysis(self.model, features)
            print("FLOPs:", flops.total())
            print("=" * 64)
            break


class MultiTaskTrainManager:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_name: List[str],
        lr: float,
        epochs: int,
        patience: int,
        wandb_log: bool = False,
    ):
        """Train Manager for SharedBottom, MMOE, PLE, STEM

        Args:
            model: model
            train_loader: train data loader
            val_loader: val data loader
            task_name: task name
            lr: learning rate
            epochs: epochs
            patience: patience
            wandb_log: whether to log to wandb
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = task_name
        self.epochs = epochs
        self.patience = patience
        self.best_weight = None

    def _train_a_batch(self, y: List[torch.Tensor], features: Dict[str, torch.Tensor]):
        """Train a batch

        Args:
            y: label of two tasks, the shape of tensor is (batch_size, )
            features: features, the shape of tensor is (batch_size, )
        """
        pred = self.model(features)

        batch_loss = self.model.get_l2_reg()
        for task_id in range(2):
            batch_loss += self.loss_func(pred[task_id].cpu(), y[task_id].float())

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

    def train(self):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            epoch_loss = 0
            for y_0, y_1, features in tqdm(self.train_loader, unit="batch"):
                y = [y_0, y_1]
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                epoch_loss += self._train_a_batch(y, features)

            epoch_loss /= len(self.train_loader)
            auc_val = self.evaluation(self.val_loader)
            if self.wandb_log:
                wandb.log(
                    {
                        "Loss": epoch_loss,
                        f"AUC-Val-{self.task_name[0]}": auc_val[0],
                        f"AUC-Val-{self.task_name[1]}": auc_val[1],
                    }
                )
            print(
                "Epoch:{}, Loss:{}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}".format(
                    epoch,
                    epoch_loss,
                    self.task_name[0],
                    auc_val[0],
                    self.task_name[1],
                    auc_val[1],
                )
            )

            if auc_val[0] > best_auc_score:
                earlystop_count = 0
                best_auc_score = auc_val[0]
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print("EarlyStopping count {}".format(earlystop_count))
                if earlystop_count == self.patience:
                    print("EarlyStopping at epoch {}".format(epoch))
                    break

    @torch.no_grad()
    def evaluation(self, data_loader: DataLoader):
        self.model.eval()
        y_true = [[] for _ in range(2)]
        y_hat = [[] for _ in range(2)]

        for y_0, y_1, features in data_loader:
            y = [y_0, y_1]
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model(features)
            for i in range(2):
                y_true[i].append(y[i])
                y_hat[i].append(pred[i])

        auc_score = []
        for i in range(2):
            y = torch.cat(y_true[i])
            pred = torch.cat(y_hat[i])
            auc_score.append(roc_auc_score(y.int(), pred.cpu()))

        return auc_scor

    def count_params(self):
        trainable_params_num, total_params_num = 0, 0
        for _, params in self.model.named_parameters():
            total_params_num += params.numel()
            if params.requires_grad:
                trainable_params_num += params.numel()
        print("=" * 64)
        print("Total params: {}".format(total_params_num))
        print("Trainable params: {}".format(trainable_params_num))
        print("=" * 64)

    def compute_cost(self):
        self.count_params()
        for _, _, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            flops = FlopCountAnalysis(self.model, features)
            print("FLOPs:", flops.total())
            print("=" * 64)
            break


class SparseSharingTrainManager(MultiTaskTrainManager):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mask_path: str,
        task_name: List[str],
        lr: float,
        epochs: int,
        patience: int,
        wandb_log: bool = False,
    ):
        """Train Manager for SparseSharing

        Args:
            model: model
            train_loader: train data loader
            val_loader: val data loader
            mask_path: mask path
            task_name: task name
            lr: learning rate
            epochs: epochs
            patience: patience
            wandb_log: whether to log to wandb
        """
        super().__init__(
            model, train_loader, val_loader, task_name, lr, epochs, patience, wandb_log
        )
        self.all_mask = torch.load(mask_path)

    def _train_a_batch(self, y, features):
        """Train a batch

        Args:
            y: label of two tasks, the shape of tensor is (batch_size, )
            features: features, the shape of tensor is (batch_size, )
        """
        for task_id in range(len(y)):
            cur_mask = self.all_mask[task_id]
            weights = copy.deepcopy(self.model.shared_bottom.state_dict())

            for name, param in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * cur_mask[name]).to(
                        self.device
                    )

            pred = self.model(features, task_id)
            loss_r = (
                self.loss_func(pred.cpu(), y[task_id].float()) + self.model.get_l2_reg()
            )
            self.optimizer.zero_grad()
            loss_r.backward()

            for name, p in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    grad_tensor = p.grad.data.cpu().numpy()
                    p.grad.data = torch.from_numpy(grad_tensor * cur_mask[name]).to(
                        self.device
                    )
            self.model.shared_bottom.load_state_dict(weights)
            self.optimizer.step()

    @torch.no_grad()
    def evaluation(self, data_loader: DataLoader):
        self.model.eval()
        y_true = [[] for _ in range(2)]
        y_hat = [[] for _ in range(2)]

        for task_id in range(2):
            weights = copy.deepcopy(self.model.shared_bottom.state_dict())
            cur_mask = self.all_mask[task_id]

            for name, param in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * cur_mask[name]).to(
                        self.device
                    )

            for y_0, y_1, features in data_loader:
                y = [y_0, y_1]
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                pred = self.model(features, task_id)
                y_true[task_id].append(y[task_id])
                y_hat[task_id].append(pred)

            self.model.shared_bottom.load_state_dict(weights)

        auc_score = []
        for task_id in range(2):
            y = torch.cat(y_true[task_id])
            pred = torch.cat(y_hat[task_id])
            auc_score.append(roc_auc_score(y.int(), pred.cpu()))

        return auc_score

    def compute_cost(self):
        self.count_params(self.model)
        for name in self.all_mask[0]:
            a = (1 - self.all_mask[0][name]) * (1 - self.all_mask[1][name])
            print("No training required:", a.sum())
        for _, _, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            flops = FlopCountAnalysis(self.model, (features, 0))
            print("FLOPs:", flops.total() * 2)
            break


class CSRecTrainManager(SparseSharingTrainManager):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mask_path: str,
        task_name: List[str],
        lr: float,
        epochs: int,
        patience: int,
        wandb_log: bool = False,
    ):
        """Train Manager for CSRec

        Args:
            model: model
            train_loader: train data loader
            val_loader: val data loader
            mask_path: mask path
            task_name: task name
            lr: learning rate
            epochs: epochs
            patience: patience
            wandb_log: whether to log to wandb
        """
        super().__init__(
            model,
            train_loader,
            val_loader,
            mask_path,
            task_name,
            lr,
            epochs,
            patience,
            wandb_log,
        )
        shared_mask = {}
        for name in self.all_mask[0]:
            shared_mask[name] = self.all_mask[0][name] * self.all_mask[1][name]
        self.contrastive_mask = {}
        for name, mask in self.all_mask[0].items():
            p = np.random.random(mask.shape)
            self.contrastive_mask[name] = np.where(p > 1, 0, 1 - shared_mask[name])

    def _train_a_batch(self, y, features):
        """Train a batch

        Args:
            y: label of two tasks, the shape of tensor is (batch_size, )
            features: features, the shape of tensor is (batch_size, )
        """
        for task_id in range(2):
            cur_mask = self.all_mask[task_id]
            weights = copy.deepcopy(self.model.shared_bottom.state_dict())

            for name, param in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(
                        tensor * self.contrastive_mask[name]
                    ).to(self.device)

            pred = self.model(features, task_id)
            loss_r_hat = -self.loss_func(pred.cpu(), y[task_id].float())
            self.optimizer.zero_grad()
            loss_r_hat.backward()

            grads = {}
            for name, p in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    tensor = p.grad.data.cpu().numpy()
                    grads[name] = torch.from_numpy(
                        tensor * self.contrastive_mask[name]
                    ).to(self.device)

            self.model.shared_bottom.load_state_dict(weights)

            for name, param in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    tensor = param.data.cpu().numpy()
                    param.data = torch.from_numpy(tensor * cur_mask[name]).to(
                        self.device
                    )

            pred = self.model(features, task_id)
            loss_r = (
                self.loss_func(pred.cpu(), y[task_id].float()) + self.model.get_l2_reg()
            )
            self.optimizer.zero_grad()
            loss_r.backward()

            for name, p in self.model.shared_bottom.named_parameters():
                if "weight" in name:
                    grad_tensor = p.grad.data.cpu().numpy()
                    p.grad.data = (
                        torch.from_numpy(grad_tensor * cur_mask[name]).to(self.device)
                        + grads[name]
                    )
            self.model.shared_bottom.load_state_dict(weights)
            self.optimizer.step()


class MPTRecTrainManager(MultiTaskTrainManager):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        env_ids: torch.Tensor,
        task_name: List[str],
        lr: float,
        batch_size: int,
        epochs: int,
        patience: int,
        gen_coe: float,
        env_coe: float,
        clustering_interval: int,
        wandb_log: bool = False,
    ):
        """Train Manager for MPTRec

        Args:
            model: model
            train_loader: train data loader
            val_loader: val data loader
            env_ids: environment ids
            task_name: task name
            lr: learning rate
            batch_size: batch size
            gen_coe: coefficient of generalization loss
            env_coe: coefficient of environment loss
            epochs: epochs
            patience: patience
            wandb_log: whether to log to wandb
        """
        super().__init__(
            model, train_loader, val_loader, task_name, lr, epochs, patience, wandb_log
        )
        self.env_ids = env_ids
        self.env_loss_func = nn.NLLLoss()
        self.batch_size = batch_size
        self.gen_coe = gen_coe
        self.env_coe = env_coe
        self.clustering_interval = clustering_interval

        self.gen_loss_0_list = []
        self.gen_loss_1_list = []
        self.fused_loss_0_list = []
        self.fused_loss_1_list = []
        self.env_loss_list = []

    def train(self):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            gen_loss_0_sum = 0
            gen_loss_1_sum = 0
            fused_loss_0_sum = 0
            fused_loss_1_sum = 0
            env_loss_sum = 0

            tepoch = tqdm(self.train_loader, unit="batch")
            for step, (y_0, y_1, features) in enumerate(tepoch):
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                p = float(step + (epoch - 1) * self.batch_size) / float(
                    self.epochs * self.batch_size
                )
                alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
                output = self.model(features, alpha)

                batch_env_ids = self.env_ids[
                    self.batch_size * step : self.batch_size * (step + 1)
                ]
                gen_loss_0 = self.loss_func(output["gen_preds"][0].cpu(), y_0.float())
                gen_loss_1 = self.loss_func(output["gen_preds"][1].cpu(), y_1.float())
                fused_loss_0 = self.loss_func(
                    output["fused_preds"][0].cpu(), y_0.float()
                )
                fused_loss_1 = self.loss_func(
                    output["fused_preds"][1].cpu(), y_1.float()
                )
                env_loss = self.env_loss_func(output["env_pred"].cpu(), batch_env_ids)
                loss = (
                    fused_loss_0
                    + fused_loss_1
                    + self.gen_coe * (gen_loss_0 + gen_loss_1)
                    + self.env_coe * env_loss
                    + self.model.get_l2_reg()
                )

                gen_loss_0_sum += gen_loss_0
                gen_loss_1_sum += gen_loss_1
                fused_loss_0_sum += fused_loss_0
                fused_loss_1_sum += fused_loss_1
                env_loss_sum += env_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            gen_loss_0_avg = env_loss_sum / len(self.train_loader)
            gen_loss_1_avg = env_loss_sum / len(self.train_loader)
            fused_loss_0_avg = env_loss_sum / len(self.train_loader)
            fused_loss_1_avg = env_loss_sum / len(self.train_loader)
            env_loss_avg = env_loss_sum / len(self.train_loader)

            if self.wandb_log:
                wandb.log(
                    {
                        "gen_loss_0": gen_loss_0_avg,
                        "gen_loss_1": gen_loss_1_avg,
                        "fused_loss_0": fused_loss_0_avg,
                        "fused_loss_1": fused_loss_1_avg,
                        "env_loss": env_loss_avg,
                    }
                )
            print(
                "gen_loss_0:{:.4f}, gen_loss_1:{:.4f}, fuse_loss_0:{:.4f}, fuse_loss_1:{:.4f}, env_loss:{:.4f}".format(
                    gen_loss_0_avg,
                    gen_loss_1_avg,
                    fused_loss_0_avg,
                    fused_loss_1_avg,
                    env_loss_avg,
                )
            )

            self.gen_loss_0_list.append(gen_loss_0_avg.item())
            self.gen_loss_1_list.append(gen_loss_1_avg.item())
            self.fused_loss_0_list.append(fused_loss_0_avg.item())
            self.fused_loss_1_list.append(fused_loss_1_avg.item())
            self.env_loss_list.append(env_loss_avg.item())

            if epoch % clustering_interval == 0:
                self.env_ids = self.cluster()

            auc_train = self.evaluation(self.train_loader)
            auc_val = self.evaluation(self.val_loader)
            if sum(auc_val[0]) > best_auc_score:
                earlystop_count = 0
                best_auc_score = sum(auc_val[0])
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print("EarlyStopping count {}".format(earlystop_count))
                if earlystop_count == self.patience:
                    print("EarlyStopping at epoch {}".format(epoch))
                    break

    @torch.no_grad()
    def cluster(self):
        self.model.eval()
        loss_func = nn.BCELoss(reduction="none")
        new_env_list = []

        for y_0, y_1, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.cluster_predict(features)
            loss_0 = loss_func(pred[0].cpu(), y_0.float())
            loss_1 = loss_func(pred[1].cpu(), y_1.float())
            loss = torch.stack([loss_0, loss_1], dim=1)
            new_env_list.append(torch.argmin(loss, dim=1))
        all_new_env = torch.cat(new_env_list, dim=0)

        env_diff = self.env_ids - all_new_env
        diff_num = torch.nonzero(env_diff != 0).shape[0]
        counts = torch.unique(all_new_env, return_counts=True)
        envs = [0, 0]
        for i in range(len(counts[0])):
            envs[counts[0][i]] = counts[1][i]
        print("diff_num:{}, env_0:{}, env_1:{}".format(diff_num, envs[0], envs[1]))

        return all_new_env

    @torch.no_grad()
    def evaluation(self, data_loader: DataLoader):
        self.model.eval()
        y_true = [[] for _ in range(2)]
        y_hat = [[] for _ in range(2)]

        for y_0, y_1, features in data_loader:
            y = [y_0, y_1]
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.predict(features)
            for i in range(2):
                y_true[i].append(y[i])
                y_hat[i].append(pred[i])

        auc_score = []
        for i in range(2):
            y = torch.cat(y_true[i])
            pred = torch.cat(y_hat[i])
            auc_score.append(roc_auc_score(y.int(), pred.cpu()))

        return auc_scor
