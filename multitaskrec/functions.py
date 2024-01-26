import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append('/data/hl/MultiTask')

from multitaskrec.model import MMOE, PLE, SharedBottom, SingleTask, SparseSharing
from multitaskrec.dataset import AliCCPDataset, ByteRecDataset, CensusIncomeDataset
from multitaskrec.train import CSRecTrainManager, SparseSharingTrainManager, MultiTaskTrainManager
from config import AliCCP_Vocabulary_Size, ByteRec_Vocabulary_Size, CensusIncome_Vocabulary_Size


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
