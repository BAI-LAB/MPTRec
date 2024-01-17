import pandas as pd
from torch.utils.data import Dataset


class AliCCPDataset(Dataset):
    def __init__(self, datafile: str, data_size: int):
        """Two tasks, CTR prediction and CVR prediction.

        Args:
            datafile: path of dataset
            data_size: size of dataset, -1 means all data

        Returns: None
        """
        super(AliCCPDataset, self).__init__()
        self.feature_names = []
        self.datafile = datafile
        self.data = []
        self.data_size = data_size
        self._load_data()

    def _load_data(self):
        print("start load data from: {}".format(self.datafile))
        count = 0
        with open(self.datafile) as f:
            self.feature_names = f.readline().strip().split(",")[2:]
            for line in f:
                line = line.strip().split(",")
                line = [int(v) for v in line]
                self.data.append(line)
                count += 1
                if self.data_size > 0 and count >= self.data_size:
                    break
        print("load data {} from {} finished".format(count, self.datafile))

    def __len__(
        self,
    ):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        click = line[0]
        conversion = line[1]
        features = dict(zip(self.feature_names, line[2:]))
        return click, conversion, features


class CensusIncomeDataset(Dataset):
    def __init__(self, datafile: str):
        """Two tasks: predicting whether income exceeds $50,000 and marital status.

        Args:
            datafile: path of dataset

        Returns: None
        """
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
        self.datafile = datafile
        df = pd.read_csv(
            self.datafile,
            delimiter=",",
        )
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        income = line[40]
        marital = line[41]
        features = dict(zip(self.feature_names, line[:40]))
        return income, marital, features

    def get_label(self, idx):
        return self.data[:, 40 + idx]


class ByteRecDataset(Dataset):
    def __init__(self, datafile: str):
        """Two tasks: predicting finish and like.

        Args:
            datafile: path of dataset

        Returns: None
        """
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
        df = pd.read_csv(self.datafile, delimiter=",")
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        finish = line[6]
        like = line[7]
        # FIXME: 删掉-1，在加载数据集时就删掉duration_time这个特征
        features = dict(zip(self.feature_names, list(line[:6]) + list(line[8:-1])))
        return finish, like, features
