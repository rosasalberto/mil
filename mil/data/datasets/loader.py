"""
Datasets extracted from http://www.multipleinstancelearning.com/datasets/ 
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    
    bags_id = df[1].unique()
    bags = [df[df[1]==bag_id][df.columns.values[2:]].values.tolist() for bag_id in bags_id]
    y = df.groupby([1])[0].first().values
    
    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(bags, y, test_size=0.2, random_state=0)
    
    return (X_train, y_train), (X_test, y_test)