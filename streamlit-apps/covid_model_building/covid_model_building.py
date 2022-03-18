# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:41:23 2021

@author: diego
"""

import pandas as pd
covid = pd.read_csv('covid_bal.csv')


df = covid.copy()


# Separating X and y
X = df.drop('resultado', axis=1)
Y = df['resultado']

# Build XGB CLF model
import xgboost as xgb
from xgboost import XGBClassifier

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.8, gamma=0.1,
              learning_rate=0.1, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=1, n_estimators=150, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1.0,use_label_encoder=False, verbosity=0)
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('covid_clf.pkl', 'wb'))