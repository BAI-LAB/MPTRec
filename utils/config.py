import time

CensusIncome = {
    'seed': 2023,
    'dense_features_num': 11,
    'sparse_features_num': 29,
    'embedding_size': 4,
    'input_size': 127,
    'lr': 1e-3,
    'epochs': 30,
    'batch_size': 256,
    'patience': 5,
}

AliCpp = {
    'seed': 2023,
    'dense_features_num': 0,
    'sparse_features_num': 18,
    'embedding_size': 5,
    'input_size': 90,
    'lr': 1e-4,
    'epochs': 5,
    'batch_size': 2000,
    'patience': 5,
}

SingleTask = {
    'expert_dnn_hidden_units': (256, 128),
    'tower_dnn_hidden_units': (64, 32),
    'reg_embedding': 3e-4,
    'reg_dnn': 3e-4,
    'patience': 5,
    'model_save_path': '/home/hl/PYCHARM/MultiTask/weight/singletask.pt',
}

SharedBottom = {
    'num_tasks': 2,
    'expert_dnn_hidden_units': (256, 128),
    'tower_dnn_hidden_units': (64, 32),
    'reg_embedding': 3e-4,
    'reg_dnn': 3e-4,
    'model_save_path': '/home/hl/PYCHARM/MultiTask/weight/sharedbottom.pt',
}

MMOE = {
    'num_tasks': 2,
    'num_experts': 3,
    'expert_dnn_hidden_units': (256, 128),
    'tower_dnn_hidden_units': (64, 32),
    'reg_embedding': 3e-4,
    'reg_dnn': 3e-4,
    'model_save_path': '/home/hl/PYCHARM/MultiTask/weight/mmoe.pt',
}

PLE = {
    'num_tasks': 2,
    'shared_expert_num': 1,
    'specific_expert_num': 1,
    'num_levels': 2,
    'expert_dnn_hidden_units': (256,),
    'tower_dnn_hidden_units': (64,),
    'reg_embedding': 3e-4,
    'reg_dnn': 3e-4,
    'model_save_path': '/home/hl/PYCHARM/MultiTask/weight/ple.pt',
}

InvChar = {
    'num_tasks': 2,
    'expert_dnn_hidden_units': (256, 128),
    'tower_dnn_hidden_units': (64, 32),
    'fused_coe': 1,
    'invariant_coe': 3e-4,
    'env_coe': 1e-3,
    'reg_embedding': 1e-3,
    'reg_dnn': 3e-5,
    'reg_classifier': 0,
    'model_save_path': '/home/hl/PYCHARM/MultiTask/invchar.pt',
}

CensusIncome_Vocabulary_Size = {
    'class_worker': 9,
    'det_ind_code': 52,
    'det_occ_code': 47,
    'education': 17,
    'hs_college': 3,
    'major_ind_code': 24,
    'major_occ_code': 15,
    'race': 5,
    'hisp_origin': 10,
    'sex': 2,
    'union_member': 3,
    'unemp_reason': 6,
    'full_or_part_emp': 8,
    'tax_filer_stat': 6,
    'region_prev_res': 6,
    'state_prev_res': 51,
    'det_hh_fam_stat': 38,
    'det_hh_summ': 8,
    'mig_chg_msa': 10,
    'mig_chg_reg': 9,
    'mig_move_reg': 10,
    'mig_same': 3,
    'mig_prev_sunbelt': 4,
    'fam_under_18': 5,
    'country_father': 43,
    'country_mother': 43,
    'country_self': 43,
    'citizenship': 5,
    'vet_question': 3
}

AliCCP_Vocabulary_Size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

ByteRec_Vocabulary_Size = {
    'uid': 70711, 
    'user_city': 396, 
    'item_id': 3687157, 
    'author_id': 778113, 
    'item_city': 456, 
    'channel': 5, 
    'music_id': 82840, 
    'device': 71681
}
