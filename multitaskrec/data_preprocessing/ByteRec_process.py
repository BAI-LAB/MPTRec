import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def process(data_path: str, write_dir: str):
    """data description can be found in https://www.biendata.xyz/competition/icmechallenge2019/

    Args:
        data_path: the path of the original data
        write_dir: the dir of the processed data

    Returns: None
    """
    data = pd.read_csv(
        data_path,
        sep="\t",
        names=[
            "uid",
            "user_city",
            "item_id",
            "author_id",
            "item_city",
            "channel",
            "finish",
            "like",
            "music_id",
            "device",
            "time",
            "duration_time",
        ],
    )
    data.drop(labels="time", axis=1, inplace=True)
    sparse_features = [
        "uid",
        "user_city",
        "item_id",
        "author_id",
        "item_city",
        "channel",
        "music_id",
        "device",
    ]
    # TODO：去掉duration_time这个特征
    dense_features = ["duration_time"]

    # Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    train_data, other_data = train_test_split(data, test_size=0.2, random_state=2023)
    val_data, test_data = train_test_split(other_data, test_size=0.5, random_state=2023)

    train_data.to_csv(f"{write_dir}/train.gz", index=False, compression="gzip")
    val_data.to_csv(f"{write_dir}/val.gz", index=False, compression="gzip")
    test_data.to_csv(f"{write_dir}/test.gz", index=False, compression="gzip")
    
    print("Byte-Rec data processing finished!")


if __name__ == "__main__":
    process(
        data_path="dataset/Byte-Rec/final_track2_train.txt",
        write_dir="dataset/Byte-Rec",
    )
