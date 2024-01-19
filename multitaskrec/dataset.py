import pandas as pd
from torch.utils.data import Dataset


class AliCCPDataset(
    Dataset
):  
    def __init__(self, datafile, data_size=-1):
        '''Three tasks, namely CTR prediction, CVR prediction and BSI prediction
        
        Args:
        
        Returns:
        '''
        super(AliCCPDataset, self).__init__()
        self.feature_names = []
        self.datafile = datafile
        self.data_size = data_size
        self.data = []
        self._load_data()

    def _load_data(self):
        print("start load data from: {}".format(self.datafile))
        count = 0
        with open(self.datafile) as f:
            self.feature_names = f.readline().strip().split(",")[2:]
            for line in f:
                line = line.strip().split(",")
                line = [int(v) for v in line]

                # TODO 处理最后一个label
                if line[-1] == 2:
                    line[-1] = 0
                else:
                    line[-1] = 1
                self.data.append(line)
                
                count += 1
                if self.data_size > -1:
                    if count >= self.data_size:
                        break
        self.feature_names = self.feature_names[1:-1]
        print("load data {} from {} finished".format(count, self.datafile))

    def __len__(
        self,
    ):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        click = line[0]
        conversion = line[1]
        business_scenario_information = line[-1]
        features = dict(zip(self.feature_names, line[3:-1]))
        return click, conversion, business_scenario_information, features


class CensusIncomeDataset(Dataset):
    def __init__(self, datafile, new_task):
        self.feature_names = [
            "age",
            "class_worker",
            "det_ind_code",
            "det_occ_code",
            "education",
            "wage_per_hour",
            "hs_college",
            "major_ind_code",
            "major_occ_code",
            "race",
            "hisp_origin",
            "sex",
            "union_member",
            "unemp_reason",
            "full_or_part_emp",
            "capital_gains",
            "capital_losses",
            "stock_dividends",
            "tax_filer_stat",
            "region_prev_res",
            "state_prev_res",
            "det_hh_fam_stat",
            "det_hh_summ",
            "instance_weight",
            "mig_chg_msa",
            "mig_chg_reg",
            "mig_move_reg",
            "mig_same",
            "mig_prev_sunbelt",
            "num_emp",
            "fam_under_18",
            "country_father",
            "country_mother",
            "country_self",
            "citizenship",
            "own_or_self",
            "vet_question",
            "vet_benefits",
            "weeks_worked",
            "year",
        ]

        df = pd.read_csv(
            datafile,
            delimiter=",",
        )
        self.feature_names.remove(new_task)
        if new_task == "education":
            df[f"label_{new_task}"] = df[new_task].apply(lambda x: 1 if x == 9 else 0)
        elif new_task == "sex":
            df[f"label_{new_task}"] = df[new_task]
        elif new_task == "race":
            df[f"label_{new_task}"] = df[new_task].apply(lambda x: 1 if x == 0 else 0)
        else:
            raise ValueError(f"Invalid new task: {new_task}")
        df.pop(new_task)
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        income = line[-3]
        marital = line[-2]
        new_task = line[-1]
        features = dict(zip(self.feature_names, line[:-3]))
        return income, marital, new_task, features


class ByteRecDataset(
    Dataset
):  # Three tasks: predicting finish, like, and duration time
    def __init__(self, datafile):
        self.feature_names = [
            "uid",
            "user_city",
            "item_id",
            "author_id",
            "item_city",
            "channel",
            "music_id",
            "device",
        ]
        self.datafile = datafile
        df = pd.read_csv(self.datafile, delimiter=",", index_col=None)
        df["duration_time"] = df["duration_time"].apply(
            lambda x: 1 if x > 0.000143 else 0
        )  # Videos longer than 10 seconds are labeled as 1
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        finish = line[6]
        like = line[7]
        duration_time = line[-1]
        features = dict(zip(self.feature_names, list(line[:6]) + list(line[8:-1])))
        return finish, like, duration_time, features
