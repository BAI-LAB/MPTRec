import numpy as np
from fvcore.nn import FlopCountAnalysis


def count_params(model):
    trainable_params_num = 0
    total_params_num = 0
    for name, params in model.named_parameters():
        print(name, params.size())
        total_params_num += params.numel()
        if params.requires_grad:
            trainable_params_num += params.numel()
    print("="*64)
    print('Total params: {}'.format(total_params_num))
    print('Trainable params: {}'.format(trainable_params_num))
    print("-"*64)


# ANCHOR Print table of zeros and non-zeros count
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


def compute_cost_0(model, train_loader):
    device = next(model.parameters()).device
    count_params(model)
    for _, _, _, features in train_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        flops = FlopCountAnalysis(model, features)
        print('FLOPs:', flops.total())
        break


def compute_cost_1(model, all_mask, train_loader):
    device = next(model.parameters()).device
    count_params(model)
    for name in all_mask[0]:
        a = (1 - all_mask[0][name]) * (1 - all_mask[1][name])
        print('No training required:', a.sum())
    for _, _, _, features in train_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        flops = FlopCountAnalysis(model, features)
        print('FLOPs:', flops.total() * 3)
        break


def compute_cost_2(mptrec, newtask, train_loader):
    # Computing the time and space cost of the model
    device = next(mptrec.parameters()).device
    count_params(newtask)
    for _, _, _, features in train_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, invariant_rep, variant_reps, env_embeddings = mptrec.get_infos(features)
        flops = FlopCountAnalysis(newtask, (dnn_input, invariant_rep, variant_reps, env_embeddings))
        print('FLOPs:', flops.total())
        break