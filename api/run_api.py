#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:53:00 2020

@author: nguyentiendung
"""

import flask
import pickle
import numpy as np
import pandas as pd
import math

from flask import request
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_log_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, chi2


UPLOAD_FOLDER = '/path/to/the/uploads'
    
def feature_selection_transformation(X, y, target, ordinal_feature):
    # check for numeric features train_data
    num_features = []
    cat_features = []
    for feature in X:
        if X[feature].dtypes == np.int or X[feature].dtypes == np.float:
            num_features.append(feature)
        else:
            cat_features.append(feature)
    
    X = X.reset_index()
    # impute using only numerical features
    imp = IterativeImputer(max_iter = 10, random_state = 42)
    imp.fit(X[num_features])
    X[num_features] = imp.transform(X[num_features])
    
    # impute using only categorical features
    imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    X[cat_features] = imp.fit_transform(X[cat_features].astype(str))
     
    # split dataframe into numeric and categorical data
    X_num = X.drop(cat_features, axis = 1)
    X_cat = X.drop(num_features, axis = 1)
    
    # saleprice correlation matrix
    k_num = round(len(X_num.columns) / 2)
    corrmat = X_num.corr()
    X_num_fs = corrmat.nlargest(k_num, target)[target].index
    
    # check for multicollinearity
    # if two features are strongly correlated with each other (>= 0.7) 
    # the feature with the lower correlation with the target variable is dropped
    multicorr = {}
    k = len(corrmat)
    for feature in corrmat:
        i = 1
        if feature != target:
            while i < k - 1:
                if corrmat[feature][i] >= 0.7 and feature != corrmat.index[i]:
                    multicorr[feature] = corrmat.index[i], corrmat[feature][i]
                i = i + 1
        
    # delete duplicates
    corr_scores = []
    for feature in list(multicorr.keys()):
        if multicorr[feature][1] in corr_scores:
            del multicorr[feature]
        else:
            corr_scores.append(multicorr[feature][1])
            
    # remove the feature with the lower correlation coefficient (pearson)
    dropped_features = [] 
    for feature1, feature2 in multicorr.items():
        if corrmat[target][feature1] < corrmat[target][feature2[0]]:
            dropped_features.append(feature1)
        else:
            dropped_features.append(feature2[0])

    # drop the features from X_num dataframe
    for feature in X_num:
        if feature in dropped_features:
            X_num = X_num.drop(feature, axis = 1) 
    X_num.drop(X_num.columns.difference(X_num_fs), 1, inplace = True)
    
    # encode ordinal features (dummy variables)
    if ordinal_feature in X_num:
        ord_data = [ordinal_feature]
        X_num = pd.get_dummies(X_num, columns = ord_data, drop_first = True)
    
    # encode categorical features
    enc = OrdinalEncoder()
    enc.fit(X_cat)
    X_cat_enc = enc.transform(X_cat)
    
    # feature selection on categorical data
    k_cat = round(len(X_cat.columns) / 2)
    fs = SelectKBest(f_regression, k_cat)
    fs.fit(X_cat_enc, y) # save!!
    X_cat_fs = fs.transform(X_cat_enc)
    X_cat_enc = pd.DataFrame(X_cat_fs)
    
    # concatenate numerical and categorical features
    df_cat = pd.DataFrame(X_cat_enc, index = list(range(len(X.index))))
    df_num = pd.DataFrame(X_num, index = list(range(len(X.index))))
    X = pd.concat([df_cat, df_num], axis = 1, sort = False)
    X = X.drop([target], axis = 1)
    return X

def predict_randomforestclass(X, y, path):
    # build model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    rf = RandomForestClassifier(n_estimators = 100, max_depth = None) 
    rf.fit(X_train, y_train)
    
    # quantifying the quality of prediction
    y_predict = rf.predict(X_test)
    acc_score = accuracy_score(y_test, y_predict)
    
    # save model on disk
    with open(path, "wb") as file:
        pickle.dump(rf, file)
    
    ret_stmt = 'Accuracy Score: ' + str(acc_score)
    return ret_stmt

def predict_randomforestregress(X, y, path):
    # build model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    rf = RandomForestRegressor(n_estimators = 800, min_samples_split = 2, min_samples_leaf = 1, 
      max_features = 'log2', max_depth = 70, bootstrap = False)
    rf.fit(X_train, y_train)
    
    # quantify quality of prediction
    y_predict = rf.predict(X_test)
    r_2_score = r2_score(y_test, y_predict)
    rmsle = math.sqrt(mean_squared_log_error(y_test, y_predict))
    
    ret_stmt = 'R^2 Score: ' + str(r_2_score) + '\n' + 'RMSLE: ' + str(rmsle)
    
    # save model on disk
    with open(path, "wb") as file:
        pickle.dump(rf, file)

    return ret_stmt

# app definition
app = flask.Flask(__name__)
app.config['DEBUG'] = True
    
@app.route('/predict', methods=['POST'])
def predict():
    # retrieve data
    data = request.get_json(force = True)
    train_data = pd.DataFrame(data)
    
    # take arguments
    target = request.args['target']
    del_feature = request.args.get('del_feature')
    ordinal_feature = request.args.get('ordinal_feature')
    model_type = request.args.get('model_type')
    path = request.args['path']
    
    # select features and target variable
    features = list(train_data)
    X = train_data[features]
    y = train_data[target]
    
    # delete features using uri arguments, only works for 1 argument
    if del_feature is not None:
        X = X.drop(del_feature, axis = 1) 
    
    feature_engineering = feature_selection_transformation(X, y, target, ordinal_feature)
    if model_type is None:
        if train_data[target].dtypes == np.object:
            prediction = predict_randomforestclass(feature_engineering, y, path)
            return prediction
        elif train_data[target].dtypes == np.float or train_data[target].dtypes == np.int:
            prediction = predict_randomforestregress(feature_engineering, y, path)
            return prediction
    elif model_type == 'classifier':
        prediction = predict_randomforestclass(feature_engineering, y, path)
        return prediction
    elif model_type == 'regressor':
        prediction = predict_randomforestregress(feature_engineering, y, path)
        return prediction
    
app.run(debug=True)
                                