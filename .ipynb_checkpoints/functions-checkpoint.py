import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split,cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
import os
warnings.filterwarnings('ignore')

# set global random seed
rand = 3
os.environ['PYTHONHASHSEED']=str(rand)
np.random.seed(rand)

def load_raw_data():
    dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
    raw_data = pd.read_csv(dataURL)
    return raw_data

def load_compas_df():
    raw_data = load_raw_data()
    compas_df = raw_data.loc[
        (raw_data['days_b_screening_arrest'] <= 30) &
        (raw_data['days_b_screening_arrest'] >= -30) &
        (raw_data['is_recid'] != -1) &
        (raw_data['c_charge_degree'] != "O") &
        (raw_data['score_text'] != "N/A")]
    return compas_df

def load_cox_data():
    parsed_dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/cox-parsed.csv'
    parsed_data = pd.read_csv(parsed_dataURL)
    return parsed_data
