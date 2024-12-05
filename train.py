import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('application_train.csv')
data.columns = ['sk_id_curr', 'target'] + list(data.columns[2:])

test_idx = data.sk_id_curr % 10 >= 7
data_dict = dict()
data_dict['tst'] = data.loc[test_idx].reset_index(drop=True)
data_dict['tr'] = data.loc[~test_idx].reset_index(drop=True)

features = data.select_dtypes(np.number).drop(columns=['target', 'sk_id_curr']).columns

X_tr, X_tst = data_dict["tr"][features].to_numpy(), data_dict["tst"][features].to_numpy()
y_tr, y_tst = data_dict["tr"]["target"].to_numpy(), data_dict["tst"]["target"].to_numpy()

prep = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

prep.fit(X_tr)

X_tr = prep.transform(X_tr)
X_tst = prep.transform(X_tst)

# C = 1, C = 1e3
model = LogisticRegression(penalty='l2', C=1, fit_intercept=True, solver='newton-cholesky', n_jobs=1)
model.fit(X_tr, y_tr)

score = roc_auc_score(y_tst, model.predict_proba(X_tst)[:, 1])

print(score)

import json

metrics_dct = {}
metrics_dct['roc_auc_score'] = score
json.dump(metrics_dct, open('./metrics.json', 'w'))

##########################
# SAVE-LOAD using pickle #
##########################
import pickle

# save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# load
#with open('model.pkl', 'rb') as f:
#    clf2 = pickle.load(f)

#roc_auc_score(y_tst, clf2.predict_proba(X_tst)[:, 1])
