# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:41:29 2019

@author: angus

Thanks to the following kernels for help and ideas:
https://www.kaggle.com/prashantkikani/help-humanity-by-helping-robots
https://www.kaggle.com/jesucristo/1-smart-robots-most-complete-notebook
https://www.kaggle.com/gpreda/robots-need-help
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

X.fillna(0, inplace = True)
X.replace(-np.inf, 0, inplace = True)
X.replace(np.inf, 0, inplace = True)
X_sub.fillna(0, inplace = True)
X_sub.replace(-np.inf, 0, inplace = True)
X_sub.replace(np.inf, 0, inplace = True)

#------------------------------Random Forrest Model----------------------------
def plot_confusion_matrix(actual, predicted, classes, title='Confusion Matrix'):
    conf_matrix = confusion_matrix(actual, predicted)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(title, size=12)
    plt.colorbar(fraction=0.05, pad=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
        horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()


#scores = []
#for estimators in [100, 500, 1000, 5000]:

folds = StratifiedKFold(n_splits=21, shuffle=True, random_state=1964)

sub_preds_rf = np.zeros((X_sub.shape[0], 9))
oof_preds_rf = np.zeros((X.shape[0]))
scores = np.zeros(folds.n_splits)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y['surface'])):
    clf =  RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state=1964)
    clf.fit(X.iloc[trn_idx], y['surface'][trn_idx])
    oof_preds_rf[val_idx] = clf.predict(X.iloc[val_idx])
    sub_preds_rf += clf.predict_proba(X_sub) / folds.n_splits
    scores[fold_] = clf.score(X.iloc[val_idx], y['surface'][val_idx])
    print('Fold: {} score: {}'.format(fold_,clf.score(X.iloc[val_idx], y['surface'][val_idx])))
print('Avg Accuracy', scores.mean())
print('Std Accuracy', scores.std())
#scores.append(score)
    
plot_confusion_matrix(y['surface'], oof_preds_rf, le.classes_, title='Confusion Matrix')

submission = pd.read_csv('sample_submission.csv')
submission['surface'] = le.inverse_transform(sub_preds_rf.argmax(axis=1))
submission.to_csv('submission_rf.csv', index=False)

importances = clf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({'importance':importances, 'feature':features})
importance_df.sort_values('importance', inplace=True)

plt.figure(figsize=(8,30))
sns.barplot(x='importance', y='feature',data=importance_df[:])
plt.show()

#------------------------------XGBoost Model----------------------------

#scores = []
#for estimators in [100, 500, 1000, 5000]:

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1964)

sub_preds_xgb = np.zeros((X_sub.shape[0], 9))
oof_preds_xgb = np.zeros((X.shape[0]))
scores = np.zeros(folds.n_splits)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y['surface'])):
    clf =  XGBClassifier(n_jobs = -1, learning_rate=0.05, n_estimators=1000)
    clf.fit(X.iloc[trn_idx], y['surface'][trn_idx], verbose=False,
            eval_set=[(X.iloc[val_idx], y['surface'][val_idx])], early_stopping_rounds=10)
    oof_preds_xgb[val_idx] = clf.predict(X.iloc[val_idx])
    sub_preds_xgb += clf.predict_proba(X_sub) / folds.n_splits
    scores[fold_] = clf.score(X.iloc[val_idx], y['surface'][val_idx])
    print('Fold: {} score: {}'.format(fold_,clf.score(X.iloc[val_idx], y['surface'][val_idx])))
print('Avg Accuracy', scores.mean())
print('Std Accuracy', scores.std())
#scores.append(score)
    
plot_confusion_matrix(y['surface'], oof_preds_xgb, le.classes_, title='Confusion Matrix')

submission = pd.read_csv('sample_submission.csv')
submission['surface'] = le.inverse_transform(sub_preds_xgb.argmax(axis=1))
submission.to_csv('submission_xgb.csv', index=False)

