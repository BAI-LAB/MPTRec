import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, List, Optional


class MLP(nn.Module):
    def __init__(
        self, hidden_units, input_size, output_activation="relu", dropout=None
    ):
        super(MLP, self).__init__()
        hidden_units = [input_size] + list(hidden_units)
        layer_num = len(hidden_units) - 1

        self.mlp = nn.Sequential()
        for i in range(layer_num - 1):
            self.mlp.add_module(
                "linear" + str(i), nn.Linear(hidden_units[i], hidden_units[i + 1])
            )
            self.mlp.add_module("relu" + str(i), nn.ReLU())

            if dropout:
                self.mlp.add_module("dropout" + str(i), nn.Dropout(dropout[i]))

        self.mlp.add_module(
            "linear" + str(layer_num - 1), nn.Linear(hidden_units[-2], hidden_units[-1])
        )
        if output_activation == "softmax":
            self.mlp.add_module("softmax" + str(layer_num - 1), nn.Softmax())
        elif output_activation == "sigmoid":
            self.mlp.add_module("sigmoid" + str(layer_num - 1), nn.Sigmoid())
        else:
            self.mlp.add_module("relu" + str(layer_num - 1), nn.ReLU())

        if dropout:
            self.mlp.add_module("dropout" + str(layer_num - 1), nn.Dropout(dropout[-1]))

    def forward(self, x):
        return self.mlp(x)

    def get_l2_reg(self):
        loss = 0
        for layer in self.mlp:
            if type(layer) is nn.Linear:
                loss += layer.weight.norm(2).pow(2) / 2.0
        return loss


class EmbeddingNetwork(nn.Module):
    def __init__(self, feature_vocabulary, embedding_size):
        super(EmbeddingNetwork, self).__init__()
        self.feature_names = list(feature_vocabulary.keys())
        self.embedding_dict = nn.ModuleDict()
        for name, size in feature_vocabulary.items():
            emb = nn.Embedding(size, embedding_size)
            # nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        feature_embedding = [
            x[name].unsqueeze(dim=1).float()
            for name in x.keys()
            if name not in self.feature_names
        ]
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name].long())
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)
        return feature_embedding

    def get_l2_reg(self):
        loss = 0
        for name in self.embedding_dict:
            loss += (self.embedding_dict[name].weight ** 2).sum() / 2.0
        return loss


class SingleTask(nn.Module):
    def __init__(
        self,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding,
        reg_dnn,
        dropout=None,
    ):
        super(SingleTask, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.shared_dnn_hidden_units = shared_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.dropout = dropout

        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.base_network = MLP(shared_dnn_hidden_units, input_size, dropout=dropout)
        self.tower_network = MLP(
            list(tower_dnn_hidden_units) + [1], shared_dnn_hidden_units[-1], "sigmoid"
        )

    def forward(self, x):
        dnn_input = self.embedding_network(x)
        mid_output = self.base_network(dnn_input)
        final_output = self.tower_network(mid_output)
        return [final_output.squeeze()]

    def functional_forward(self, x, params):
        embedding_network = EmbeddingNetwork(
            self.feature_vocabulary, self.embedding_size
        )
        base_network = MLP(
            self.shared_dnn_hidden_units, self.input_size, dropout=self.dropout
        )
        tower_network = MLP(
            list(self.tower_dnn_hidden_units) + [1],
            self.shared_dnn_hidden_units[-1],
            "sigmoid",
        )

        embedding_network.load_state_dict(params, strict=False)
        base_network.load_state_dict(params, strict=False)
        tower_network.load_state_dict(params, strict=False)

        dnn_input = self.embedding_network(x)
        mid_output = self.base_network(dnn_input)
        final_output = self.tower_network(mid_output)
        return [final_output.squeeze()]

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = self.base_network.get_l2_reg() + self.tower_network.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn


class SharedBottom(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding,
        reg_dnn,
        dropout=None,
    ):
        super(SharedBottom, self).__init__()
        self.num_tasks = num_tasks
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.shared_bottom = MLP(shared_dnn_hidden_units, input_size, dropout=dropout)
        self.tower_networks = nn.ModuleList()

        for _ in range(num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    shared_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x):
        dnn_input = self.embedding_network(x)
        mid_output = self.shared_bottom(dnn_input)
        final_outputs = []
        for tower in self.tower_networks:
            final_outputs.append(tower(mid_output).squeeze())
        return final_outputs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = self.shared_bottom.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "tower_networks" not in name:
                param.requires_grad = False


class MMOE(nn.Module):
    def __init__(
        self,
        num_tasks,
        num_experts,
        feature_vocabulary,
        embedding_size,
        input_size,
        expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=0,
        reg_dnn=0,
        dropout=None,
    ):
        super(MMOE, self).__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.expert_networks = nn.ModuleList()
        self.tower_networks = nn.ModuleList()
        self.gate_networks = nn.ModuleList()

        for _ in range(0, num_experts):
            self.expert_networks.append(
                MLP(expert_dnn_hidden_units, input_size, dropout=dropout)
            )
        for _ in range(0, num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    expert_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )
            self.gate_networks.append(
                nn.Sequential(
                    nn.Linear(input_size, num_experts, bias=False), nn.Softmax()
                )
            )

    def forward(self, x):
        feature_embedding = self.embedding_network(x)
        expert_outs = []
        for expert in self.expert_networks:
            expert_outs.append(expert(feature_embedding))
        expert_concat = torch.stack(expert_outs, dim=2)

        gate_outs = []
        for gate in self.gate_networks:
            gate_outs.append(gate(feature_embedding))
        weighted_expert_outs = []
        for gate_out in gate_outs:
            output = torch.matmul(expert_concat, gate_out.unsqueeze(dim=2)).squeeze()
            weighted_expert_outs.append(output)

        task_outs = []
        for i in range(self.num_tasks):
            output = self.tower_networks[i](weighted_expert_outs[i])
            task_outs.append(output.squeeze())
        return task_outs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = 0
        for expert in self.expert_networks:
            loss_dnn += expert.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "tower_networks" not in name and "gate_networks" not in name:
                param.requires_grad = False


class StopGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return None


class STEM(nn.Module):
    def __init__(
        self,
        task_num: int,
        shared_expert_num: int,
        specific_expert_num: int,
        feature_vocabulary: Dict[str, int],
        embedding_size: int,
        input_size: int,
        expert_dnn_hidden_unit: List[int],
        tower_dnn_hidden_unit: List[int],
        reg_embedding: Optional[float] = None,
        reg_dnn: Optional[float] = None,
        dropout: Optional[List[float]] = None,
    ):
        """STEM initialization.

        Args:
            task_num: number of tasks
            expert_num: number of experts
            feature_vocabulary: record the range of features
            embedding_size: embedding size of features
            input_size: input size of expert network
            expert_dnn_hidden_unit: hidden units of expert network
            tower_dnn_hidden_unit: hidden units of tower network
            reg_embedding: regularization coefficient of embedding layer
            reg_dnn: regularization coefficient of dnn layer
            dropout: dropout rate of dnn layer

        Returns: None
        """
        super(STEM, self).__init__()
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.shared_embedding_network = EmbeddingNetwork(
            feature_vocabulary, embedding_size
        )
        self.shared_expert_network = MLP(
            expert_dnn_hidden_unit, input_size, dropout=dropout
        )
        self.specific_embedding_networks = nn.ModuleList()
        self.specific_expert_networks = nn.ModuleList()
        self.tower_networks = nn.ModuleList()
        self.gate_networks = nn.ModuleList()

        for _ in range(task_num):
            self.specific_embedding_networks.append(
                EmbeddingNetwork(feature_vocabulary, embedding_size)
            )
            self.specific_expert_networks.append(
                MLP(expert_dnn_hidden_unit, input_size, dropout=dropout)
            )
            self.tower_networks.append(
                MLP(
                    tower_dnn_hidden_unit + [1],
                    expert_dnn_hidden_unit[-1],
                    output_activation="sigmoid",
                )
            )
            self.gate_networks.append(
                nn.Sequential(
                    nn.Linear(
                        input_size,
                        self.specific_expert_num * self.task_num
                        + self.shared_expert_num,
                        bias=False,
                    ),
                    nn.Softmax(dim=1),
                )
            )

    def forward(self, x):
        """Forward propagation of STEM.

        Args:
            x: input features

        Returns: output of STEM
        """
        shared_feature_embedding = self.shared_embedding_network(x)
        specific_feature_embeddings = []
        for embedding in self.specific_embedding_networks:
            specific_feature_embeddings.append(embedding(x))

        shared_expert_out = self.shared_expert_network(shared_feature_embedding)
        specific_expert_outs = []
        for expert, feature_embedding in zip(
            self.specific_expert_networks, specific_feature_embeddings
        ):
            specific_expert_outs.append(expert(feature_embedding))

        gate_outs = []
        for gate, feature_embedding in zip(
            self.gate_networks, specific_feature_embeddings
        ):
            gate_outs.append(gate(feature_embedding + shared_feature_embedding))

        weighted_expert_outs = []
        for i, gate_out in enumerate(gate_outs):
            specific_expert_outs = [
                StopGradient.apply(expert_out) if i != j else expert_out
                for j, expert_out in enumerate(specific_expert_outs)
            ]
            expert_concat = torch.stack(
                specific_expert_outs + [shared_expert_out], dim=2
            )
            output = torch.matmul(expert_concat, gate_out.unsqueeze(dim=2)).squeeze()
            weighted_expert_outs.append(output)

        task_outs = []
        for i, tower in enumerate(self.tower_networks):
            output = tower(weighted_expert_outs[i])
            task_outs.append(output.squeeze())
        return task_outs

    def get_l2_reg(self):
        """Calculate the l2 regularization of STEM.

        Args: None

        Returns: l2 regularization of STEM
        """
        loss_embedding = self.shared_embedding_network.get_l2_reg()
        for embedding in self.specific_embedding_networks:
            loss_embedding += embedding.get_l2_reg()

        loss_dnn = self.shared_expert_network.get_l2_reg()
        for expert in self.specific_expert_networks:
            loss_dnn += expert.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()

        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def get_reps(self):
        shared_feature_embedding = self.shared_embedding_network(x)
        specific_feature_embeddings = []
        for embedding in self.specific_embedding_networks:
            specific_feature_embeddings.append(embedding(x))

        shared_expert_out = self.shared_expert_network(shared_feature_embedding)
        specific_expert_outs = []
        for expert, feature_embedding in zip(
            self.specific_expert_networks, specific_feature_embeddings
        ):
            specific_expert_outs.append(expert(feature_embedding))

        return shared_expert_out, specific_expert_outs

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "shared" in name:
                param.requires_grad = False


class CGC(nn.Module):
    def __init__(
        self,
        num_tasks,
        input_size,
        specific_expert_num,
        shared_expert_num,
        expert_dnn_hidden_units,
        dropout,
    ):
        super(CGC, self).__init__()
        self.num_tasks = num_tasks
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.shared_expert_networks = nn.ModuleList()
        self.specific_expert_networks = nn.ModuleList()
        self.shared_gate = MLP(
            (num_tasks * specific_expert_num + shared_expert_num,),
            input_size,
            "softmax",
        )
        self.specific_gates = nn.ModuleList()

        # build task-shared expert layer
        for _ in range(shared_expert_num):
            expert_network = MLP(expert_dnn_hidden_units, input_size, "relu", dropout)
            self.shared_expert_networks.append(expert_network)

        # build task-specific expert layer
        for _ in range(num_tasks):
            for _ in range(specific_expert_num):
                expert_network = MLP(
                    expert_dnn_hidden_units, input_size, "relu", dropout
                )
                self.specific_expert_networks.append(expert_network)

        # task-specific gate
        for _ in range(num_tasks):
            gate_network = MLP(
                (specific_expert_num + shared_expert_num,), input_size, "softmax"
            )
            self.specific_gates.append(gate_network)

    def forward(self, inputs, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_expert_networks[
                    i * self.specific_expert_num + j
                ](inputs[i])
                specific_expert_outputs.append(specific_expert_output)
        shared_expert_outputs = []
        for i in range(self.shared_expert_num):
            shared_expert_output = self.shared_expert_networks[i](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)

        cgc_outs = []
        for i in range(self.num_tasks):
            cur_experts = (
                specific_expert_outputs[
                    i * self.specific_expert_num : (i + 1) * self.specific_expert_num
                ]
                + shared_expert_outputs
            )
            expert_concat = torch.stack(cur_experts, dim=2)
            gate_out = self.specific_gates[i](inputs[i])  # gate[i] for task input[i]
            gate_out = torch.unsqueeze(gate_out, -1)
            gate_mul_expert = torch.matmul(expert_concat, gate_out).squeeze()
            cgc_outs.append(gate_mul_expert)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_experts = (
                specific_expert_outputs + shared_expert_outputs
            )  # all the expert include task-specific expert and task-shared expert
            expert_concat = torch.stack(cur_experts, dim=2)
            gate_out = self.shared_gate(inputs[-1])  # gate for shared task input
            gate_out = torch.unsqueeze(gate_out, -1)
            gate_mul_expert = torch.matmul(expert_concat, gate_out).squeeze()
            cgc_outs.append(gate_mul_expert)

        return cgc_outs

    def get_l2_reg(self):
        loss = 0
        for expert in self.shared_expert_networks:
            loss += expert.get_l2_reg()
        for expert in self.specific_expert_networks:
            loss += expert.get_l2_reg()
        return loss

    def get_reps(self, inputs):
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_expert_networks[
                    i * self.specific_expert_num + j
                ](inputs[i])
                specific_expert_outputs.append(specific_expert_output)
        shared_expert_outputs = []
        for i in range(self.shared_expert_num):
            shared_expert_output = self.shared_expert_networks[i](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)

        return shared_expert_outputs, specific_expert_outputs


class PLE(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_expert_num=1,
        specific_expert_num=1,
        num_levels=2,
        expert_dnn_hidden_units=(256,),
        tower_dnn_hidden_units=(64,),
        reg_embedding=0,
        reg_dnn=0,
        dropout=None,
    ):
        super(PLE, self).__init__()
        self.num_tasks = num_tasks
        self.num_levels = num_levels
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.cgc_networks = nn.ModuleList()
        self.tower_networks = nn.ModuleList()

        self.cgc_networks.append(
            CGC(
                num_tasks=num_tasks,
                input_size=input_size,
                shared_expert_num=shared_expert_num,
                specific_expert_num=specific_expert_num,
                expert_dnn_hidden_units=expert_dnn_hidden_units,
                dropout=dropout,
            )
        )

        for _ in range(num_levels - 1):
            self.cgc_networks.append(
                CGC(
                    num_tasks=num_tasks,
                    input_size=expert_dnn_hidden_units[-1],
                    shared_expert_num=shared_expert_num,
                    specific_expert_num=specific_expert_num,
                    expert_dnn_hidden_units=expert_dnn_hidden_units,
                    dropout=dropout,
                )
            )

        for _ in range(num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    expert_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x):
        feature_embedding = self.embedding_network(x)
        ple_inputs = [feature_embedding] * (
            self.num_tasks + 1
        )  # [task1, task2, ... taskn, shared task]
        ple_outputs = []

        for i in range(self.num_levels):
            if i == self.num_levels - 1:  # the last level
                ple_outputs = self.cgc_networks[i](inputs=ple_inputs, is_last=True)
            else:
                ple_outputs = self.cgc_networks[i](inputs=ple_inputs, is_last=False)
                ple_inputs = ple_outputs

        task_outs = []
        for i in range(self.num_tasks):
            task_out = self.tower_networks[i](ple_outputs[i])
            task_outs.append(task_out.squeeze())

        return task_outs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = 0
        for cgc in self.cgc_networks:
            loss_dnn += cgc.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "embedding_network" in name or "shared_expert_networks" in name:
                param.requires_grad = False

    def get_reps(self, x):
        feature_embedding = self.embedding_network(x)
        ple_inputs = [feature_embedding] * (
            self.num_tasks + 1
        )  # [task1, task2, ... taskn, shared task]
        ple_outputs = []

        for i in range(self.num_levels):
            if i == self.num_levels - 1:
                generic_rep, proprietary_rep = self.cgc_networks[i].get_reps(
                    inputs=ple_inputs
                )
            else:
                ple_outputs = self.cgc_networks[i](inputs=ple_inputs, is_last=False)
                ple_inputs = ple_outputs

        return generic_rep[0], proprietary_rep


class SparseSharing(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding,
    ):
        super(SparseSharing, self).__init__()
        self.num_tasks = num_tasks
        self.reg_embedding = reg_embedding
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.shared_bottom = MLP(shared_dnn_hidden_units, input_size)
        self.tower_networks = nn.ModuleList()

        for _ in range(num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    shared_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x, task_id=0):
        dnn_input = self.embedding_network(x)
        mid_output = self.shared_bottom(dnn_input)
        final_output = self.tower_networks[task_id](mid_output)
        return final_output.squeeze()

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        return self.reg_embedding * loss_embedding


class EnvClassifier(nn.Module):
    def __init__(self):
        super(EnvClassifier, self).__init__()

    def forward(self, invariant_preferences):
        raise NotImplementedError

    def get_l2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_l1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class LinearLogSoftMaxEnvClassifier(EnvClassifier):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_l1_reg(self) -> torch.Tensor:
        return (
            torch.norm(self.linear_map.weight, 1) / self.elements_num
            + torch.norm(self.linear_map.bias, 1) / self.bias_num
        )

    def get_l2_reg(self) -> torch.Tensor:
        return (
            torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num
            + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num
        )

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class Expert(nn.Module):
    def __init__(self, num_tasks, input_dim, expert_dnn_hidden_units, dropout=None):
        super(Expert, self).__init__()
        self.num_tasks = num_tasks
        self.shared_network = MLP(expert_dnn_hidden_units, input_dim, "relu", dropout)
        self.specific_networks = nn.ModuleList()

        for i in range(num_tasks):
            self.specific_networks.append(
                MLP(expert_dnn_hidden_units, input_dim, "relu", dropout)
            )

    def forward(self, dnn_input):
        invariant_rep = self.shared_network(dnn_input)
        variant_reps = []
        for i in range(self.num_tasks):
            variant_reps.append(self.specific_networks[i](dnn_input))
        return invariant_rep, variant_reps

    def get_l2_reg(self):
        loss = 0
        loss += self.shared_network.get_l2_reg()
        for network in self.specific_networks:
            loss += network.get_l2_reg()
        return loss


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MPTRec(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        expert_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding=0,
        reg_dnn=0,
        dropout=None,
        device=None,
    ):
        super(MPTRec, self).__init__()
        self.num_tasks = num_tasks
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.device = device
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.shared_expert_network = MLP(expert_dnn_hidden_units, input_size, "relu", dropout)
        self.specific_expert_networks = nn.ModuleList()
        self.env_embedding_network = nn.Embedding(num_tasks, expert_dnn_hidden_units[-1])
        self.gate_networks = nn.ModuleList()
        self.env_classifier = LinearLogSoftMaxEnvClassifier(
            expert_dnn_hidden_units[-1], num_tasks
        )
        self.tower_networks = nn.ModuleList()

        for _ in range(num_tasks):
            self.specific_expert_networks.append(
                MLP(expert_dnn_hidden_units, input_size, "relu", dropout)
            )
            self.gate_networks.append(
                nn.Sequential(nn.Linear(input_size, 2, bias=False), nn.Softmax())
            )
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    expert_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x, alpha=1):
        dnn_input = self.embedding_network(x)
        gen_rep = self.shared_expert_network(dnn_input)

        gen_preds = []
        for i in range(self.num_tasks):
            output = self.tower_networks[i](gen_rep)
            gen_preds.append(output.squeeze())

        gate_outs = []
        for gate in self.gate_networks:
            gate_outs.append(gate(dnn_input))

        fused_preds = []
        for i in range(self.num_tasks):
            spec_rep = self.specific_expert_networks[i](dnn_input)
            env_embedding = self.env_embedding_network(torch.tensor(i).to(self.device))
            env_aware_rep = spec_rep * env_embedding
            all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
            fused_rep = torch.matmul(all_reps, gate_outs[i].unsqueeze(dim=2)).squeeze()
            output = self.tower_networks[i](fused_rep)
            fused_preds.append(output.squeeze())

        rev_gen_rep = ReverseLayerF.apply(gen_rep, alpha)
        env_pred = self.env_classifier(rev_gen_rep)

        return {
            "gen_preds": gen_preds,
            "fused_preds": fused_preds,
            "env_pred": env_pred,
        }

    def predict(self, x):
        output = self.forward(x)
        return output["fused_preds"]

    def cluster_predict(self, x):
        # TODO: 尝试仅使用通用表征的预测结果做聚类预测
        return self.predict(x)

    def get_infos(self, x):
        dnn_input = self.embedding_network(x)
        gen_rep = self.shared_expert_network(dnn_input)

        spec_reps, env_embs = []
        for i in range(self.num_tasks):
            spec_reps.append(self.specific_expert_networks[i](dnn_input))
            env_embs.append(self.env_embedding_network(torch.tensor(i).to(self.device)))

        return {
            "dnn_input": dnn_input,
            "gen_rep": gen_rep,
            "spec_reps": spec_reps,
            "env_embs": env_embs,
        }

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = self.shared_expert_network.get_l2_reg()
        for expert in self.specific_expert_networks:
            loss_dnn += expert.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn


class NewTask(nn.Module):
    def __init__(
        self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None
    ):
        super(NewTask, self).__init__()
        self.reg_dnn = reg_dnn
        self.device = device
        self.temperature = 150
        self.env_embedding_network = nn.Embedding(1, rep_dim)
        self.projection_network = nn.Sequential(
            nn.Linear(input_size, rep_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(rep_dim // 2, rep_dim, bias=False),
            nn.LayerNorm(rep_dim),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, 2, bias=False), nn.Softmax()
        )
        self.gate_network_2 = nn.Sequential(
            nn.Linear(input_size, 2, bias=False), nn.Softmax()
        )
        self.tower_network = MLP(
            list(tower_dnn_hidden_units) + [1],
            input_size=rep_dim,
            output_activation="sigmoid",
        )

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        exist_env_embs = torch.cat(env_embs)
        new_env_emb = self.env_embedding_network(torch.tensor(0).to(self.device))

        H_out = self.projection_network(dnn_input)
        W = torch.mm(H_out, exist_env_embs.T) / self.temperature
        W = F.softmax(W, dim=-1).unsqueeze(2)

        gate_out = self.gate_network(dnn_input).unsqueeze(dim=2)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), W).squeeze()
        env_aware_rep = new_spec_rep * new_env_emb
        all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
        fused_rep = torch.matmul(all_reps, gate_out).squeeze()

        output = self.tower_network(fused_rep)
        return output.squeeze()

    def get_l2_reg(self):
        loss_dnn = self.tower_network.get_l2_reg()
        return self.reg_dnn * loss_dnn
