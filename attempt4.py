# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:16:42 2019

@author: angus
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

X = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
X_sub = pd.read_csv('X_test.csv')

#kind of useful values to keep track of
n = X['series_id'].nunique()
n_sub = X_sub['series_id'].nunique()
targets = y['surface'].unique()
n_groups = y['group_id'].nunique()
n_targets = y['surface'].value_counts().reset_index().rename(columns={'index': 'target'})

#-----------------------------Some visualisation-------------------------------
missing = X.isnull().sum() #Any missing data?
duplicates = X.duplicated().value_counts() #Any duplicate data?

sns.barplot(y='target', x='surface', data=n_targets)

#Histograms 
plt.figure(figsize=(20, 20))
for i, col in enumerate(X.columns[3:]):
    plt.subplot(4, 3, i + 1)
    sns.distplot(X[col], color='blue', bins=100, label='train')
    sns.distplot(X_sub[col], color='green', bins=100, label='sub')
    plt.legend()

#Example single measurement
plt.figure(figsize=(20, 20))
for i, col in enumerate(X.columns[3:]):
    plt.subplot(4, 3, i + 1)
    plt.plot(X.loc[X['series_id'] == 5, col])
    plt.title(col)

correlation = X.corr().iloc[2:, 2:] #correlations between features
 
df = X.merge(y, on='series_id', how='inner')

plt.figure(figsize=(26,16))
for i,col in enumerate(df.columns[3:13]):
    ax = plt.subplot(3,4,i+1)
    ax = plt.title(col)
    for surface in targets:
        surface_feature = df[df['surface'] == surface]
        sns.kdeplot(surface_feature[col], label = surface)

#------------------------------Feature engineering-----------------------------

def perform_feature_engineering(df):
    df_out = pd.DataFrame()
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))

    def mean_abs_change(x):
        return np.mean(np.abs(np.diff(x)))
    
    for col in df.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        df_out[col + '_mean'] = df.groupby(['series_id'])[col].mean()
        df_out[col + '_min'] = df.groupby(['series_id'])[col].min()
        df_out[col + '_max'] = df.groupby(['series_id'])[col].max()
        df_out[col + '_std'] = df.groupby(['series_id'])[col].std()
        df_out[col + '_mad'] = df.groupby(['series_id'])[col].mad()
        df_out[col + '_med'] = df.groupby(['series_id'])[col].median()
        df_out[col + '_skew'] = df.groupby(['series_id'])[col].skew()
        df_out[col + '_range'] = df_out[col + '_max'] - df_out[col + '_min']
        df_out[col + '_max_to_min'] = df_out[col + '_max'] / df_out[col + '_min']
        df_out[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(mean_abs_change)
        df_out[col + '_mean_change_of_abs_change'] = df.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df_out[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
        df_out[col + '_abs_min'] = df.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))
        df_out[col + '_abs_mean'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(x)))
        df_out[col + '_abs_std'] = df.groupby('series_id')[col].apply(lambda x: np.std(np.abs(x)))
        df_out[col + '_abs_avg'] = (df_out[col + '_abs_min'] + df_out[col + '_abs_max'])/2
        df_out[col + '_abs_range'] = df_out[col + '_abs_max'] - df_out[col + '_abs_min']

    return df_out

def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def fe_step0 (actual):
    
    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html
    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html
    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
        
    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)
    actual['mod_quat'] = (actual['norm_quat'])**0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']
    
    return actual

def fe_step1 (actual):
    """Quaternions to Euler Angles"""
    
    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual

X = fe_step0(X)
X = fe_step1(X)
X = perform_feature_engineering(X)

X_sub = fe_step0(X_sub)
X_sub = fe_step1(X_sub)
X_sub = perform_feature_engineering(X_sub)

correlations = X.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]

n_top_corr = correlations[correlations[0]==1.0].shape[0]
print("There are {} different features pairs with correlation factor equal to 1.0.".format(n_top_corr))

n_top_corr = correlations[correlations[0]>0.9999].shape[0]
print("There are {} different features pairs with correlation factor greater than 0.9999.".format(n_top_corr))

"""
drop_features = list(correlations.head(n_top_corr)['level_0'].unique())
X = X.drop(drop_features,axis=1)
X_sub = X_sub.drop(drop_features,axis=1)
"""

corr = X.corr()
fig, ax = plt.subplots(1,1,figsize=(16,16))
sns.heatmap(corr,  xticklabels=False, yticklabels=False)
plt.show()

le = LabelEncoder()
y['surface'] = le.fit_transform(y['surface'])

ohe = OneHotEncoder()
ohe_y = ohe.fit_transform(np.array(y['surface']).reshape(-1,1)).toarray()

X.fillna(0, inplace = True)
X.replace(-np.inf, 0, inplace = True)
X.replace(np.inf, 0, inplace = True)
X_sub.fillna(0, inplace = True)
X_sub.replace(-np.inf, 0, inplace = True)
X_sub.replace(np.inf, 0, inplace = True)


#----------------------------------DL Model------------------------------------

def create_model():
    model = Sequential()
    model.add(Dense(200, activation='relu'))#, input_shape=(128,10)))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def plot_training(model, name='model', save_loss=False, save_acc=False):
    #Plot accuracy
    plt.figure(figsize=(14,8))
    plt.plot(model.history.history['val_categorical_accuracy'][:], label='validation')
    plt.plot(model.history.history['categorical_accuracy'][:], label = 'train')
    plt.title('categorical_accuracy')
    plt.legend()
    if save_acc == True:
        plt.savefig(name + '_acc.png')
    plt.show()
    
    #plot loss
    plt.figure(figsize=(14,8))
    plt.plot(model.history.history['val_loss'][:], label='validation')
    plt.plot(model.history.history['loss'][:], label = 'train')
    plt.title('loss (categorical crossentropy)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    if save_loss == True:
        plt.savefig(name + '_loss.png')
    plt.show()

DL_model = create_model()

DL_model.fit(np.array(X), ohe_y, epochs=40)

plot_training(DL_model)

sub_preds_dl = DL_model.predict(X_sub)

submission = pd.read_csv('sample_submission.csv')
submission['surface'] = le.inverse_transform(sub_preds_dl.argmax(axis=1))
submission.to_csv('submission_dl.csv', index=False)



#Below doesn't work atm
folds = StratifiedKFold(n_splits=21, shuffle=True, random_state=1964)

sub_preds_dl = np.zeros((X_sub.shape[0], 9))
oof_preds_dl = np.zeros((X.shape[0]))
scores = np.zeros(folds.n_splits)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, ohe_y)):
    clf =  create_model()
    clf.fit(X.iloc[trn_idx], ohe_y[trn_idx])
    oof_preds_dl[val_idx] = clf.predict(X.iloc[val_idx])
    sub_preds_dl += clf.predict_proba(X_sub) / folds.n_splits
    scores[fold_] = clf.score(X.iloc[val_idx], y['surface'][val_idx])
    print('Fold: {} score: {}'.format(fold_,clf.score(X.iloc[val_idx], y['surface'][val_idx])))
print('Avg Accuracy', scores.mean())
print('Std Accuracy', scores.std())
