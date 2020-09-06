#encoding: utf-8
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

def read_df(cfg):

    train_df = pd.read_csv(f'{cfg.DATASETS.ROOT_DIR}/train.csv')
    train_df.drop_duplicates(subset=['Patient','Weeks'], keep = False, inplace = True)

    test_df = pd.read_csv(f'{cfg.DATASETS.ROOT_DIR}/test.csv')

    sub_df = pd.read_csv(f'{cfg.DATASETS.ROOT_DIR}/sample_submission.csv')

    sub_df[['Patient','Weeks']] = sub_df.Patient_Week.str.split("_",expand = True)
    sub_df =  sub_df[['Patient','Weeks','Confidence', 'Patient_Week']]
    sub_df = sub_df.merge(test_df.drop('Weeks', axis = 1), on = "Patient")

    train_df['Source'] = 'train'
    sub_df['Source'] = 'test'

    data_df = train_df.append([sub_df])
    data_df.reset_index(inplace = True)
    data_df.head()

    data_df = get_baseline_week(data_df)
    data_df = get_baseline_FVC(data_df)

    num_attribs = ['FVC', 'Percent', 'Age', 'baselined_week', 'base_FVC']
    cat_attribs = ['Sex', 'SmokingStatus']

    own_MinMaxColumnScaler(data_df, num_attribs)
    own_OneHotColumnCreator(data_df, cat_attribs)

    train_df = data_df.loc[data_df.Source == 'train']
    sub = data_df.loc[data_df.Source == 'test']

    return train_df, sub


def own_MinMaxColumnScaler(df, columns):
    """Adds columns with scaled numeric values to range [0, 1]
    using the formula X_scld = (X - X.min) / (X.max - X.min)"""
    for col in columns:
        new_col_name = col + '_scld'
        col_min = df[col].min()
        col_max = df[col].max()        
        df[new_col_name] = (df[col] - col_min) / ( col_max - col_min )

def own_OneHotColumnCreator(df, columns):
    """OneHot Encodes categorical features. Adds a column for each unique value per column"""
    for col in columns:
        for value in df[col].unique():
            df[value] = (df[col] == value).astype(int)


def get_baseline_week(df):
    # make a copy to not change original df    
    _df = df.copy()
    # ensure all Weeks values are INT and not accidentaly saved as string
    _df['Weeks'] = _df['Weeks'].astype(int)
    # as test data is containing all weeks, 
    _df.loc[_df.Source == 'test','min_week'] = np.nan
    _df["min_week"] = _df.groupby('Patient')['Weeks'].transform('min')
    _df['baselined_week'] = _df['Weeks'] - _df['min_week']
    
    return _df

def get_baseline_FVC(df):
    # same as above
    _df = df.copy()
    base = _df.loc[_df.Weeks == _df.min_week]
    base = base[['Patient','FVC']].copy()
    base.columns = ['Patient','base_FVC']
    
    # add a row which contains the cumulated sum of rows for each patient
    base['nb'] = 1
    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
    
    # drop all except the first row for each patient (= unique rows!), containing the min_week
    base = base[base.nb == 1]
    base.drop('nb', axis = 1, inplace = True)
    
    # merge the rows containing the base_FVC on the original _df
    _df = _df.merge(base, on = 'Patient', how = 'left')    
    _df.drop(['min_week'], axis = 1)
    
    return _df