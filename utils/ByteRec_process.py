import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# data description can be found in https://www.biendata.xyz/competition/icmechallenge2019/
data = pd.read_csv('/home/hl/MultiTask/data/ByteRec/final_track2_train.txt', sep='\t',
                    names=["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like",
                            "music_id", "device", "time", "duration_time"])

sparse_features = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
dense_features = ["duration_time"]

target = ['finish', 'like']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=2023)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=2023)
train_data.to_csv('/home/hl/MultiTask/data/ByteRec/train.gz', index=False, compression='gzip')
val_data.to_csv('/home/hl/MultiTask/data/ByteRec/val.gz', index=False, compression='gzip')
test_data.to_csv('/home/hl/MultiTask/data/ByteRec/test.gz', index=False, compression='gzip')
