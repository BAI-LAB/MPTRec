import numpy as np
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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


def process0(data_path, write_path, test_size=None, random_state=None):  # 40个特征，2个标签，分别是income和marital
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    data = pd.read_csv(
        data_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)
    columns = data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code','education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital']]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    if test_size:
        val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        val_data.to_csv(write_path+'val/{}.gz'.format(random_state), index=False, compression='gzip')
        test_data.to_csv(write_path+'test/{}.gz'.format(random_state), index=False, compression='gzip')
    else:
        data.to_csv(write_path+'test.gz', index=False, compression='gzip')


def process1(data_path, write_path, test_size=None, random_state=None):  # 39个特征，3个标签，分别是income，marital和education
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    data = pd.read_csv(
        data_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    data['label_education'] = data['education'].apply(lambda x: 1 if x == ' Bachelors degree(BA AB BS)' else 0)
    data.drop(labels=['income_50k', 'marital_stat', 'education'], axis=1, inplace=True)
    columns = data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital', 'label_education']]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    if test_size:
        val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        val_data.to_csv(write_path+'val/#{}.gz'.format(random_state), index=False, compression='gzip')
        test_data.to_csv(write_path+'test/#{}.gz'.format(random_state), index=False, compression='gzip')
    else:
        data.to_csv(write_path, index=False, compression='gzip')


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