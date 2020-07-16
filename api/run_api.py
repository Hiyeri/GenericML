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
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif


UPLOAD_FOLDER = '/path/to/the/uploads'
    
def regress_feature_selection_transformation(X, y, target, ordinal_feature, route):
    
    def train_num_feature_selection(X_num):
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
        
        dropped_features = []
        # remove the feature with the lower correlation coefficient (pearson) 
        for feature1, feature2 in multicorr.items():
            if corrmat[target][feature1] < corrmat[target][feature2[0]]:
                dropped_features.append(feature1)
            else:
                dropped_features.append(feature2[0])
    
        # drop the features from X_num dataframe
        for feature in X_num:
            if feature in dropped_features:
                X_num = X_num.drop(feature, axis = 1)
        drop_corr_features = X_num.columns.difference(X_num_fs)
        X_num.drop(X_num.columns.difference(X_num_fs), 1, inplace = True)
        
        return (X_num, dropped_features, drop_corr_features)
    
    def train_cat_feature_selection(X_cat, X_cat_enc):
        # feature selection on categorical data
        k_cat = round(len(X_cat.columns) / 2)
        fs = SelectKBest(f_classif, k_cat)
        fs.fit(X_cat_enc, y) # save!!
        X_cat_fs = fs.transform(X_cat_enc)
        X_cat_enc = pd.DataFrame(X_cat_fs)
        
        return (X_cat_enc, fs)
    
    def predict_num_feature_selection(X_num):
        with open('fs_values.pkl', 'rb') as file:
            drop_corr_features, drop_multicoll_features = pickle.load(file)[:2]
            
        X_num = X_num.drop(drop_multicoll_features, axis = 1) 
        X_num.drop(drop_corr_features, 1, inplace = True)
        
        return X_num
    
    def predict_cat_feature_selection(X_cat_enc):
        with open('fs_values.pkl', 'rb') as file:
            selected_cat_features, dummy = pickle.load(file)[2:4]
            
        X_cat_fs = selected_cat_features.transform(X_cat_enc)
        X_cat_enc = pd.DataFrame(X_cat_fs)
        
        return X_cat_enc
    
    # split features
    num_features = []
    cat_features = []
    for feature in X:
        if X[feature].dtypes == np.int or X[feature].dtypes == np.float:
            num_features.append(feature)
        else:
            cat_features.append(feature)
    
    X = X.reset_index()

    # impute using only numerical features
    if num_features:
        imp = IterativeImputer(max_iter = 10, random_state = 42)
        imp.fit(X[num_features])
        X[num_features] = imp.transform(X[num_features])
        X_num = X.drop(cat_features, axis = 1)
    
    # impute using only categorical features
    if cat_features:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        X[cat_features] = imp.fit_transform(X[cat_features].astype(str))
        X_cat = X.drop(num_features, axis = 1)
        
    # get column count
    if num_features:
        num_shape = X_num.shape[1]
    
    if cat_features:
        cat_shape = X_cat.shape[1]
    
    # feature selection numerical features
    if route == '/train':
        if num_features and not num_shape <= 10:
            X_num, drop_multicoll_features, drop_corr_features = train_num_feature_selection(X_num)
        else:
            drop_multicoll_features = []
            drop_corr_features = []
    elif route == '/predict':
        if num_features and not num_shape <= 10:
            X_num = predict_num_feature_selection(X_num)
        
    # encode ordinal features (dummy variables)
    if ordinal_feature is not None and ordinal_feature in X_num:
        ord_data = [ordinal_feature]
        X_num = pd.get_dummies(X_num, columns = ord_data, drop_first = True)

    # encode categorical features
    if cat_features:
        enc = OrdinalEncoder()
        enc.fit(X_cat)
        X_cat_enc = enc.transform(X_cat)
    
    # feature selection caterorical features
    if route == '/train':
        if cat_features and not cat_shape <= 10:
            X_cat_enc, selected_cat_features = train_cat_feature_selection(X_cat, X_cat_enc)
        else:
            selected_cat_features = None
    elif route == '/predict':
        if cat_features and not cat_shape <= 10:
            X_cat_enc = predict_cat_feature_selection(X_cat_enc)
    
    # concatenate numerical and categorical features
    if cat_features and num_features:
        df_cat = pd.DataFrame(X_cat_enc, index = list(range(len(X.index))))
        df_num = pd.DataFrame(X_num, index = list(range(len(X.index))))
        X = pd.concat([df_cat, df_num], axis = 1, sort = False)
        if route == '/train':
            X = X.drop([target], axis = 1)
    elif cat_features:
        X = pd.DataFrame(X_cat_enc)
    elif num_features:
        X = pd.DataFrame(X_num)
    
    if route == '/train':
        # serialize feature selection values
        fs_values = [drop_corr_features, drop_multicoll_features, selected_cat_features, ordinal_feature, target]
        with open('fs_values.pkl', 'wb') as file:
            pickle.dump(fs_values, file)
        
    return X

def class_feature_selection_transformation(X, y, target, ordinal_feature, route):
    
    def num_feature_selection(X_num):
        # feature selection on numerical data
        k_num = round(len(X_num.columns) / 2)
        fs = SelectKBest(f_classif, k_num)
        fs.fit(X_num, y) # save!!
        X_num_fs = fs.transform(X_num)
        return X_num_fs
        
    def cat_feature_selection(X_cat, X_cat_enc):
        # feature selection on categorical data
        k_cat = round(len(X_cat.columns) / 2)
        fs = SelectKBest(chi2, k_cat)
        fs.fit(X_cat_enc, y) # save!!
        X_cat_fs = fs.transform(X_cat_enc)
        X_cat_enc = pd.DataFrame(X_cat_fs)
        return X_cat_enc
    
    # split features
    num_features = []
    cat_features = []
    for feature in X:
        if X[feature].dtypes == np.int or X[feature].dtypes == np.float:
            num_features.append(feature)
        else:
            cat_features.append(feature)
    
    X = X.reset_index()
    
    # impute using only numerical features
    if num_features:
        imp = IterativeImputer(max_iter = 10, random_state = 42)
        imp.fit(X[num_features])
        X[num_features] = imp.transform(X[num_features])
        X_num = X.drop(cat_features, axis = 1)
    
    # impute using only categorical features
    if cat_features:
        imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        X[cat_features] = imp.fit_transform(X[cat_features].astype(str))
        X_cat = X.drop(num_features, axis = 1)
        
    # get column count
    if num_features:
        num_shape = X_num.shape[1]
    
    if cat_features:
        cat_shape = X_cat.shape[1]
    
    # drop target
    if target in X_num:
        X_num = X_num.drop(target, axis = 1)
    if target in X_cat:
        X_cat = X_cat.drop(target, axis = 1)
    
    if num_features and not num_shape <= 10:
        X_num = num_feature_selection(X_num)
    
    # encode ordinal features (dummy variables)
    if ordinal_feature is not None:
        ord_data = [ordinal_feature]
        X_num = pd.get_dummies(X_num, columns = ord_data, drop_first = True)
    
    if cat_features:
        enc = OrdinalEncoder()
        enc.fit(X_cat)
        X_cat_enc = enc.transform(X_cat)
    
    if cat_features and not cat_shape <= 10:
        X_cat_enc = cat_feature_selection(X_cat, X_cat_enc)
    
    # concatenate numerical and categorical features
    if num_features and cat_features:
        df_cat = pd.DataFrame(X_cat_enc, index = list(range(len(X.index))))
        df_num = pd.DataFrame(X_num, index = list(range(len(X.index))))
        X = pd.concat([df_cat, df_num], axis = 1, sort = False)
    elif cat_features:
        X = pd.DataFrame(X_cat_enc)
    elif num_features:
        X = pd.DataFrame(X_num)
    
    return X

def predict_randomforestclass(X, y):
    # build model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    rf = RandomForestClassifier(n_estimators = 100, max_depth = None) 
    rf.fit(X_train, y_train)
    
    # quantifying the quality of prediction
    y_predict = rf.predict(X_test)
    acc_score = accuracy_score(y_test, y_predict)
    
    # save model on disk
    with open('model.pkl', "wb") as file:
        pickle.dump(rf, file)
    
    ret_stmt = 'Accuracy Score: ' + str(acc_score)
    return ret_stmt

def predict_randomforestregress(X, y):
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
    with open('model.pkl', "wb") as file:
        pickle.dump(rf, file)

    return ret_stmt

# app definition
app = flask.Flask(__name__)
app.config['DEBUG'] = True
    
@app.route('/train', methods=['POST'])
def train():
    route = request.path
    # retrieve data
    data = request.get_json(force = True)
    train_data = pd.DataFrame(data)
    
    # take arguments
    target = request.args['target']
    del_feature = request.args.get('del_feature')
    ordinal_feature = request.args.get('ordinal_feature')
    model_type = request.args.get('model_type')
    
    # select features and target variable
    features = list(train_data)
    X = train_data[features]
    y = train_data[target]
    
    # delete features using uri arguments, only works for 1 argument
    if del_feature is not None:
        X = X.drop(del_feature, axis = 1) 
    
    if model_type is None:
        if train_data[target].dtypes == np.object:
            feature_engineering = class_feature_selection_transformation(X, y, target, ordinal_feature, route)
            prediction = predict_randomforestclass(feature_engineering, y)
            return prediction
        elif train_data[target].dtypes == np.float or train_data[target].dtypes == np.int:
            feature_engineering = regress_feature_selection_transformation(X, y, target, ordinal_feature, route)
            prediction = predict_randomforestregress(feature_engineering, y)
            return prediction
    elif model_type == 'classifier':
        feature_engineering = class_feature_selection_transformation(X, y, target, ordinal_feature, route)
        prediction = predict_randomforestclass(feature_engineering, y)
        return prediction
    elif model_type == 'regressor':
        feature_engineering = regress_feature_selection_transformation(X, y, target, ordinal_feature, route)
        prediction = predict_randomforestregress(feature_engineering, y)
        return prediction
    
@app.route('/predict', methods=['POST'])
def predict():
   route = request.path
   # retrieve data
   data = request.get_json(force = True)
   test_data = pd.DataFrame(data)

   with open('fs_values.pkl', 'rb') as file:
        ordinal_feature, target = pickle.load(file)[3:]
   with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

   feature_engineering = regress_feature_selection_transformation(test_data, None, None, ordinal_feature, route)
   y_predict = model.predict(feature_engineering)
    
   df_features = pd.DataFrame(test_data)
   df_prediction = pd.DataFrame({target: y_predict})
   output = pd.concat([df_features, df_prediction], axis = 1, sort = False)
   output.to_csv('prediction.csv', index = False)

   return "Successful"
    
app.run(debug=True)
                                