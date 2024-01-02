import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append('/data/hl/MultiTask')

from utils.models import MMOE, PLE, SharedBottom, SingleTask, SparseSharing
from utils.dataset import AliCCPDataset, ByteRecDataset, CensusIncomeDataset
from utils.train import CSRecTrainManager, SparseSharingTrainManager, TrainManager
from utils.config import AliCCP_Vocabulary_Size, ByteRec_Vocabulary_Size, CensusIncome_Vocabulary_Size


def print_nonzeros(model, print_flag=True):
    nonzero = total = 0
    for name, p in model.shared_bottom.named_parameters():
        if 'weight' in name:
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            if print_flag:
                print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) '
                      f'| total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    if print_flag:
        print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, '
              f'Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
    return round(((total - nonzero) / total), 3)


def compare_mask(all_mask_0, all_mask_1):
    similarity = {}
    for name in all_mask_0:
        sub = all_mask_0[name] - all_mask_1[name]
        nz_count = np.count_nonzero(sub)
        total = np.prod(sub.shape)
        similarity[name] = round(1 - nz_count / total, 4)
    print(similarity)


def count_prune_rate(cur_mask):
    nz_count, total_params = 0, 0
    for name in cur_mask:
        nz_count += np.count_nonzero(cur_mask[name])
        total_params += np.prod(cur_mask[name].shape)
    prune_rate = 1 - nz_count / total_params
    return round(prune_rate, 4)


def load_dataset(dataset='CensusIncome', seed=2023):
    if dataset == 'CensusIncome':
        train_dataset = CensusIncomeDataset('data/CensusIncome/train.gz')
        test_dataset = CensusIncomeDataset('data/CensusIncome/test.gz')
        val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
        train_loader = DataLoader(train_dataset, batch_size=256)
        val_loader = DataLoader(val_dataset, batch_size=256)
        test_loader = DataLoader(test_dataset, batch_size=256)
    
    elif dataset == 'AliCCP':
        train_dataset = AliCCPDataset('data/AliCpp/ctr_cvr.train', 2000000)
        val_dataset = AliCCPDataset('data/AliCpp/ctr_cvr.dev', 200000)
        test_dataset = AliCCPDataset('data/AliCpp/ctr_cvr.test', 2000000)
        train_loader = DataLoader(train_dataset, batch_size=2000)
        val_loader = DataLoader(val_dataset, batch_size=2000)
        test_loader = DataLoader(test_dataset, batch_size=2000)

    elif dataset == 'ByteRec':
        train_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/train.gz')
        val_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/val.gz')
        test_dataset = ByteRecDataset('/data/hl/MultiTask/data/ByteRec/test.gz')
        train_loader = DataLoader(train_dataset, batch_size=4000)
        val_loader = DataLoader(val_dataset, batch_size=4000)
        test_loader = DataLoader(test_dataset, batch_size=4000)

    return train_loader, val_loader, test_loader


def load_model(model='SingleTask', dataset='CensusIncome'):
    if model == 'SingleTask':
        if dataset == 'CensusIncome':
            model = SingleTask(
                feature_vocabulary=CensusIncome_Vocabulary_Size,
                embedding_size=4,
                input_size=127,
                shared_dnn_hidden_units=(256, 128),
                tower_dnn_hidden_units=(64, 32),
                reg_embedding=0,
                reg_dnn=0,
            )
        
        elif dataset == 'AliCCP':
            model = SingleTask(
                feature_vocabulary=AliCCP_Vocabulary_Size,
                embedding_size=5,
                input_size=90,
                shared_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=0,
                reg_dnn=0,
                dropout=(0.1, 0.3)
            )

        elif dataset == 'ByteRec':
            model = SingleTask(
                feature_vocabulary=ByteRec_Vocabulary_Size,
                embedding_size=4,
                input_size=32,
                shared_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=0,
                reg_dnn=0,
            )
    
    elif model == 'SharedBottom':
        if dataset == 'CensusIncome':
            model = SharedBottom(
                num_tasks=2,
                feature_vocabulary=CensusIncome_Vocabulary_Size,
                embedding_size=4,
                input_size=127,
                shared_dnn_hidden_units=(256, 128),
                tower_dnn_hidden_units=(64, 32),
                reg_embedding=3e-4,
                reg_dnn=3e-4
            )
        
        elif dataset == 'AliCCP':
            model = SharedBottom(
                num_tasks=2,
                feature_vocabulary=AliCCP_Vocabulary_Size,
                embedding_size=5,
                input_size=90,
                shared_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6,
                reg_dnn=1e-6,
                dropout=(0.1, 0.3)
            )

        elif dataset == 'ByteRec':
            model = SharedBottom(
                num_tasks=2,
                feature_vocabulary=ByteRec_Vocabulary_Size,
                embedding_size=4,
                input_size=32,
                shared_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6,
                reg_dnn=1e-6,
            )

    elif model == 'MMOE':
        if dataset == 'CensusIncome':
            model = MMOE(
                num_tasks=2,
                num_experts=3,
                feature_vocabulary=CensusIncome_Vocabulary_Size,
                embedding_size=4,
                input_size=127,
                expert_dnn_hidden_units=(256, 128),
                tower_dnn_hidden_units=(64, 32),
                reg_embedding=3e-4,
                reg_dnn=3e-4
            )

        elif dataset == 'AliCCP':
            model = MMOE(
                num_tasks=2,
                num_experts=3,
                feature_vocabulary=AliCCP_Vocabulary_Size,
                embedding_size=5,
                input_size=90,
                expert_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6,
                reg_dnn=1e-6,
                dropout=(0.1, 0.3),
            )
        
        elif dataset == 'ByteRec':
            model = MMOE(
                num_tasks=2,
                num_experts=3,
                feature_vocabulary=ByteRec_Vocabulary_Size,
                embedding_size=4,
                input_size=32,
                expert_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6,
                reg_dnn=1e-6,
            )

    elif model == 'PLE':
        if dataset == 'CensusIncome':
            model = PLE(
                num_tasks=2,
                input_size=127,
                feature_vocabulary=CensusIncome_Vocabulary_Size,
                embedding_size=4,
                shared_expert_num=1,
                specific_expert_num=1,
                num_levels=2,
                expert_dnn_hidden_units=(256, ),
                tower_dnn_hidden_units=(64, 32),
                reg_embedding=3e-4,
                reg_dnn=3e-4
            )
     
        elif dataset == 'AliCCP':
            model = PLE(
                num_tasks=2,
                input_size=90,
                feature_vocabulary=AliCCP_Vocabulary_Size,
                embedding_size=5,
                shared_expert_num=1,
                specific_expert_num=1,
                num_levels=2,
                expert_dnn_hidden_units=(128,),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6,
                reg_dnn=1e-6,
                dropout=(0.1, 0.3),
            )

        elif dataset == 'ByteRec':
            model = PLE(
                num_tasks=2,
                feature_vocabulary=ByteRec_Vocabulary_Size,
                embedding_size=4,
                input_size=32,
                shared_expert_num=1,
                specific_expert_num=1,
                num_levels=2,
                expert_dnn_hidden_units=(128,),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6,
                reg_dnn=1e-6
            )

    elif model in ['SparseSharing', 'CSRec']:
        if dataset == 'CensusIncome':
            model = SparseSharing(
                num_tasks=2,
                feature_vocabulary=CensusIncome_Vocabulary_Size,
                embedding_size=4,
                input_size=127,
                shared_dnn_hidden_units=(256, 128),
                tower_dnn_hidden_units=(64, 32),
                reg_embedding=3e-4
            )
        
        elif dataset == 'AliCCP':
            model = SparseSharing(
                num_tasks=2,
                feature_vocabulary=AliCCP_Vocabulary_Size,
                embedding_size=5,
                input_size=90,
                shared_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6
            ) 
        
        elif dataset == 'ByteRec':
            model = SparseSharing(
                num_tasks=2,
                feature_vocabulary=ByteRec_Vocabulary_Size,
                embedding_size=4,
                input_size=32,
                shared_dnn_hidden_units=(128, 64),
                tower_dnn_hidden_units=(32, 32),
                reg_embedding=1e-6
            )

    return model


def load_train_manager(model, dataset, config):
    if model in ['SingleTask', 'SharedBottom', 'MMOE', 'PLE']:
        if dataset == 'CensusIncome':
            train_manager = TrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                task_name=['Income', 'Marital'],
                lr=1e-3
            )
        
        elif dataset == 'AliCCP':
            train_manager = TrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                task_name=['CTR', 'CVR'],
                lr=1e-4
            )

        elif dataset == 'ByteRec':
            train_manager = TrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                task_name=['Finish', 'Like'],
                epochs=10,
                lr=1e-4
            )

    elif model == 'SparseSharing':
        if dataset == 'CensusIncome':
            train_manager = SparseSharingTrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                mask_path='',
                task_name=['Income', 'Marital'],
                lr=1e-3
            )
        
        elif dataset == 'AliCCP':
            train_manager = SparseSharingTrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                mask_path='',
                task_name=['CTR', 'CVR'],
                lr=1e-4
            )

        elif dataset == 'ByteRec':
            train_manager = SparseSharingTrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                mask_path='',
                task_name=['Finish', 'Like'],
                epochs=10,
                lr=1e-4
            )         

    elif model == 'CSRec':
        if dataset == 'CensusIncome':
            train_manager = CSRecTrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                mask_path='',
                task_name=['Income', 'Marital'],
                lr=1e-3
            )
        
        elif dataset == 'AliCCP':
            train_manager = CSRecTrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                mask_path='',
                task_name=['CTR', 'CVR'],
                lr=1e-4
            )

        elif dataset == 'ByteRec':
            train_manager = CSRecTrainManager(
                model=config['model'],
                train_loader=config['train_loader'],
                val_loader=config['val_loader'],
                mask_path='',
                task_name=['Finish', 'Like'],
                epochs=10,
                lr=1e-4
            )

    return train_manager