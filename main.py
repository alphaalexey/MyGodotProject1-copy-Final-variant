### Import libraries and datasets

import math

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold


SUBMIT_FOLDER = 'Results'
MATERIALS_FOLDER = 'Siberian Alfa Hack Materials'

train_df = pd.read_parquet(f'{MATERIALS_FOLDER}/train.parquet')
test_df = pd.read_parquet(f'{MATERIALS_FOLDER}/test.parquet')

types = train_df.dtypes
targets = train_df.columns[train_df.columns.str.contains('target')]
features = train_df.columns.drop(targets).drop(['id'])

print(train_df.shape)
print(test_df.shape)

### Data Cleaning

def find_anomalies(x: pd.core.series.Series) -> pd.core.series.Series:
  vc = x.value_counts().sort_index().iloc[np.r_[0:4, -4:0]]
  anomalies = vc[(450 < vc) & (vc < 505) | (950 < vc) & (vc < 1010)].index

  return x.mask(x.isin(anomalies))

cl_train = train_df.copy().drop(columns=['id']).iloc[0:300000]

indexes = features[types[features] != 'object']

cl_train[indexes] = cl_train[indexes].apply(lambda x: find_anomalies(x))

print(cl_train.shape)

### Selecting Features

letters = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

def letter_features(letters):
  letter_features = pd.Index([])
  for letter in letters:
    s = features[features.str.contains(f'_{letter}_')]
    letter_features = letter_features.union(s)
  return letter_features

cat_features = features[types[features] == 'object']
num_features = features.drop(cat_features)

city_features = features[features.str.contains(f'city')]
min_city_features = city_features.drop('index_city_code')

non_fin_features = features[features.str.contains(f'_non_fin_')]
fin_features = features[features.str.contains(f'_fin_')].drop(non_fin_features)

balance_features = features[features.str.contains(f'balance')]

ogrn_features = features[features.str.contains(f'ogrn')]

founderpres_features = features[features.str.contains(f'founderpres')]

sum_of_paym_features = features[features.str.contains(f'sum_of_paym')]

all_letter_features = letter_features(letters)

bad_time_features = features[23:27]

letter_drop = []
features_drop = ['branch_code', 'okved', 'index_city_code', 'city_type', 'channel_code']

### Training

final_train_df = cl_train.copy()
final_train_df = final_train_df.sample(frac=1)

final_train_df[cat_features] = final_train_df[cat_features].astype("category")
final_train_df = final_train_df.drop(letter_drop + features_drop, axis=1)

X = final_train_df.drop(["target_1", "target_2", "total_target"], axis=1)

y1 = final_train_df.target_1
y2 = final_train_df.target_2

model_t1 = LGBMClassifier(verbosity=-1, random_state=42)
model_t2 = LGBMClassifier(verbosity=-1, random_state=42)

skf = StratifiedKFold(n_splits=5)
for tr_i, tst_i in skf.split(X, y1):
  x1_train = X.iloc[tr_i]
  x1_val = X.iloc[tst_i]
  y1_train = y1.iloc[tr_i]
  y1_val = y1.iloc[tst_i]
  model_t1.fit(x1_train, y1_train)
  y1_pred = model_t1.predict_proba(x1_val)[:, 1]
  print(f"m_target1 ROC-AUC {roc_auc_score(y1_val, y1_pred)}")

print()
for tr_i, tst_i in skf.split(X, y2):
  x2_train = X.iloc[tr_i]
  x2_val = X.iloc[tst_i]
  y2_train = y2.iloc[tr_i]
  y2_val = y2.iloc[tst_i]
  model_t2.fit(x2_train, y2_train)
  y2_pred = model_t2.predict_proba(x2_val)[:, 1]
  print(f"m_target2 ROC-AUC {roc_auc_score(y2_val, y2_pred)}")

### Model Prediction

test = test_df.copy()
test[cat_features] = test_df[cat_features].astype("category")
test = test.drop(features_drop + letter_drop + ["id"], axis=1)

test_score_t1 = model_t1.predict_proba(test)[:, 1]
test_score_t2 = model_t2.predict_proba(test)[:, 1]

test_score = np.maximum(test_score_t1, test_score_t2)

submit = pd.read_csv(f'{MATERIALS_FOLDER}/sample_submission.csv')

submit['score'] = test_score
submit.head()

submit.to_csv(f'{SUBMIT_FOLDER}/result.csv', index=False)
