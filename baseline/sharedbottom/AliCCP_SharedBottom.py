import sys
import torch

import warnings

import numpy as np

from torch.utils.data import DataLoader


sys.path.append('/home/hl/MultiTask/')


from multitaskrec.train import MultiTaskTrainManager

from multitaskrec.model import SharedBottom

from multitaskrec.dataset import AliCCPDataset

from config import AliCCP_Vocabulary_Size


warnings.filterwarnings('ignore')



def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
        

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

    device = torch.device("cuda:3")

    model.to(device)


    # from fvcore.nn import FlopCountAnalysis

    # from utils.functions import count_params

    # count_params(model)

    # for _, _, features in train_loader:

    #     for key in features.keys():

    #         features[key] = features[key].to(device)

    #     flops = FlopCountAnalysis(model, features)

    #     print('FLOPs:', flops.total() / 1e6)

    #     break


    train_manager = MultiTaskTrainManager(
        model=model,
        train_loader=train_loader,

        val_loader=val_loader,

        task_name=['CTR', 'CVR'],

        lr=1e-4
    )

    train_manager.train(2)


    model.load_state_dict(train_manager.best_weight)

    auc_test = train_manager.evaluation(test_loader, 2)

    print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}'.format(auc_test[0], auc_test[1]))

    

if __name__ == '__main__':

    train_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.train', -1)

    val_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.dev', -1)

    test_dataset = AliCCPDataset('/home/hl/MultiTask/data/AliCpp/ctr_cvr.test', -1)

    train_loader = DataLoader(train_dataset, batch_size=2000)

    val_loader = DataLoader(val_dataset, batch_size=2000)

    test_loader = DataLoader(test_dataset, batch_size=2000)


    for seed in [1688723512, 1688723740, 1688738016, 1688749593, 1688762746]:
        main()

