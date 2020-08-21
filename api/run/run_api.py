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
import os.path
import json
import requests
import shutil
import threading

from flask import request, Flask, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
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
from sklearn.linear_model import LinearRegression, LogisticRegression
    
def regress_feature_selection_transformation(X, y, target, ordinal_feature, route, path):
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
        directory = path + '/fs_values.pkl'
        with open(directory, 'rb') as file:
            drop_multicoll_features, drop_corr_features = pickle.load(file)[:2]
            
        X_num = X_num.drop(drop_multicoll_features, axis = 1) 
        X_num.drop(drop_corr_features, 1, inplace = True)
        
        return X_num
    
    def predict_cat_feature_selection(X_cat_enc):
        directory = path + '/fs_values.pkl'
        with open(directory, 'rb') as file:
            selected_cat_features, dummy = pickle.load(file)[2:4]
            
        X_cat_fs = selected_cat_features.transform(X_cat_enc)
        X_cat_enc = pd.DataFrame(X_cat_fs)
        
        return X_cat_enc
    
    # split features
    num_features = []
    cat_features = []
    X_num = pd.DataFrame()
    X_cat = pd.DataFrame()
    
    for feature in X:
        if X[feature].dtypes == np.int or X[feature].dtypes == np.float:
            num_features.append(feature)
        else:
            cat_features.append(feature)
    
    # impute using only numerical features
    if num_features and not all(elem == target for elem in num_features):
        X = X.reset_index(drop = True)
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
    if num_features and not all(elem == target for elem in num_features):
        num_shape = X_num.shape[1]
    else:
        num_shape = 0
    
    if cat_features:
        cat_shape = X_cat.shape[1]
    else:
        cat_shape = 0
    
    # feature selection numerical features
    if route == '/train':
        if num_features and not all(elem == target for elem in num_features) and not num_shape <= 10:
            X_num, drop_multicoll_features, drop_corr_features = train_num_feature_selection(X_num)
        else:
            drop_multicoll_features = []
            drop_corr_features = []
            selected_num_features = None
    elif route == '/predict':
        if num_features and not num_shape <= 10:
            X_num = predict_num_feature_selection(X_num)
        
    # encode ordinal features (dummy variables)
    if ordinal_feature and all(item in X_num for item in ordinal_feature):
        for feature in ordinal_feature:
            ord_data = [feature]
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
    if cat_features and num_features and not all(elem == target for elem in num_features):
        df_cat = pd.DataFrame(X_cat_enc, index = list(range(len(X.index))))
        df_num = pd.DataFrame(X_num, index = list(range(len(X.index))))
        X = pd.concat([df_cat, df_num], axis = 1, sort = False)
    elif cat_features:
        X = pd.DataFrame(X_cat_enc)
    elif num_features:
        X = pd.DataFrame(X_num)
    
    if route == '/train':
        # serialize feature selection values
        fs_values = [drop_multicoll_features, drop_corr_features, selected_cat_features, ordinal_feature, target]
        directory = path + '/fs_values.pkl'
        with open(directory, 'wb') as file:
            pickle.dump(fs_values, file)
        
    return X

def class_feature_selection_transformation(X, y, target, ordinal_feature, route, path):
    def train_num_feature_selection(X_num):
        # feature selection on numerical data
        k_num = round(len(X_num.columns) / 2)
        fs = SelectKBest(f_classif, k_num)
        fs.fit(X_num, y) # save!!
        X_num_fs = fs.transform(X_num)
        return (X_num_fs, fs)
        
    def train_cat_feature_selection(X_cat, X_cat_enc):
        # feature selection on categorical data
        k_cat = round(len(X_cat.columns) / 2)
        fs = SelectKBest(chi2, k_cat)
        fs.fit(X_cat_enc, y) # save!!
        X_cat_fs = fs.transform(X_cat_enc)
        X_cat_enc = pd.DataFrame(X_cat_fs)
        return (X_cat_enc, fs)
    
    def predict_num_feature_selection(X_num):
        directory = path + '/fs_values.pkl'
        with open(directory, 'rb') as file:
            selected_num_features, dummy = pickle.load(file)[0:2]
            
        X_num_fs = selected_num_features.transform(X_num)
        X_num = pd.DataFrame(X_num_fs)
        
        return X_num
    
    def predict_cat_feature_selection(X_cat_enc):
        directory = path + '/fs_values.pkl'
        with open(directory, 'rb') as file:
            dummy, selected_cat_features = pickle.load(file)[0:2]
            
        X_cat_fs = selected_cat_features.transform(X_cat_enc)
        X_cat_enc = pd.DataFrame(X_cat_fs)
        
        return X_cat_enc
    
    # split features
    num_features = []
    cat_features = []
    X_num = pd.DataFrame()
    X_cat = pd.DataFrame()
    
    for feature in X:
        if X[feature].dtypes == np.int or X[feature].dtypes == np.float:
            num_features.append(feature)
        else:
            cat_features.append(feature)
    
    # impute using only numerical features
    if num_features:
        X = X.reset_index(drop = True)
        imp = IterativeImputer(max_iter = 10, random_state = 42)
        imp.fit(X[num_features])
        X[num_features] = imp.transform(X[num_features])
        X_num = X.drop(cat_features, axis = 1)
    
    # impute using only categorical features
    if cat_features and not all(elem == target for elem in cat_features):
        imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        X[cat_features] = imp.fit_transform(X[cat_features].astype(str))
        X_cat = X.drop(num_features, axis = 1)
        
    # get column count
    if not X_num.empty:
        num_shape = X_num.shape[1]
    else:
        num_shape = 0
    
    if not X_cat.empty:
        cat_shape = X_cat.shape[1]
    else:
        cat_shape = 0
    
    # drop target
    if target in X_num:
        X_num = X_num.drop(target, axis = 1)
    if target in X_cat:
        X_cat = X_cat.drop(target, axis = 1)
    
    # feature selection on num_features
    if route == '/train':
        if num_features and not num_shape <= 10:
            X_num, selected_num_features = train_num_feature_selection(X_num)
        else:
            selected_num_features = None
    if route == '/predict':
        if num_features and not num_shape <= 10:
            X_num = predict_num_feature_selection(X_num)
    
    # encode ordinal features (dummy variables)
    if ordinal_feature:
        for feature in ordinal_feature:
            ord_data = [feature]
            X_num = pd.get_dummies(X_num, columns = ord_data, drop_first = True)
    
    if cat_features:
        enc = OrdinalEncoder()
        enc.fit(X_cat)
        X_cat_enc = enc.transform(X_cat)
    
    # feature selection on cat_features
    if route == '/train':
        if cat_features and not cat_shape <= 10:
            X_cat_enc, selected_cat_features = train_cat_feature_selection(X_cat, X_cat_enc)
        else:
            selected_cat_features = None
    if route == '/predict':
        if cat_features and not cat_shape <= 10:
            X_cat_enc = predict_cat_feature_selection(X_cat_enc)
    
    # concatenate numerical and categorical features
    if num_features and cat_features:
        df_cat = pd.DataFrame(X_cat_enc, index = list(range(len(X.index))))
        df_num = pd.DataFrame(X_num, index = list(range(len(X.index))))
        X = pd.concat([df_cat, df_num], axis = 1, sort = False)
    elif cat_features:
        X = pd.DataFrame(X_cat_enc)
    elif num_features:
        X = pd.DataFrame(X_num)
    
    if route == '/train':
        # serialize feature selection values
        fs_values = [selected_num_features, selected_cat_features, ordinal_feature, target]
        directory = path + '/fs_values.pkl'
        with open(directory, 'wb') as file:
            pickle.dump(fs_values, file)
            
    return X

def predict_randomforestregress(X, y, target):
    # build model
    if target in X:
        X = X.drop(target, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    rf = RandomForestRegressor(n_estimators = 800, min_samples_split = 2, min_samples_leaf = 1, 
      max_features = 'log2', max_depth = 70, bootstrap = False)
    rf.fit(X_train, y_train)
    
    # quantify quality of prediction
    y_predict = rf.predict(X_test)
    r_2_score = abs(r2_score(y_test, y_predict))
    model = (rf, r_2_score)
    
    return model

def predict_linearregress(X, y, target):
    # build model
    if target in X:
        X = X.drop(target, axis = 1)
    X_scaled = preprocessing.scale(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 42)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # quantify quality of prediction
    y_predict = reg.predict(X_test)
    r_2_score = abs(r2_score(y_test, y_predict))
    
    model = (reg, r_2_score)
    return model

def predict_randomforestclass(X, y, target):
    if target in X:
        X = X.drop(target, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    rf = RandomForestClassifier(n_estimators = 100, max_depth = None) 
    rf.fit(X_train, y_train)
    
    # quantifying the quality of prediction
    y_predict = rf.predict(X_test)
    acc_score = abs(accuracy_score(y_test, y_predict))
    
    model = (rf, acc_score)
    return model

def predict_logisticregress(X, y, target):
    if target in X:
        X = X.drop(target, axis = 1)
    X_scaled = preprocessing.scale(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 42)
    
    log_reg = LogisticRegression(random_state = 42) 
    log_reg.fit(X_train, y_train)
    
    # quantifying the quality of prediction
    y_predict = log_reg.predict(X_test)
    acc_score = abs(accuracy_score(y_test, y_predict))
    
    model = (log_reg, acc_score)
    return model

def build_model(path, target, ordinal_feature, route, model_type):
    def predict_regress(feature_engineering, y, target, path):
        rf, rf_r2 = predict_randomforestregress(feature_engineering, y, target)
        reg, reg_r2 = predict_linearregress(feature_engineering, y, target)
        
        best_score = max([rf_r2, reg_r2])
        directory_model = path + '/model.pkl'
        
        # save best model on disk
        if best_score == rf_r2:
            with open(directory_model, 'wb') as file:
                pickle.dump(rf, file)
                model = 'RandomForestRegressor'
                scoring_param = 'r2 score'
                return (model, scoring_param, rf_r2)
            
        elif best_score == reg_r2:
            with open(directory_model, 'wb') as file:
                pickle.dump(reg, file)
                model = 'LinearRegression'
                scoring_param = 'r2 score'
                return (model, scoring_param, reg_r2)
            
    def predict_class(feature_engineering, y, target, path):
        rf, rf_acc = predict_randomforestclass(feature_engineering, y, target)
        log_reg, log_acc = predict_logisticregress(feature_engineering, y, target)
        
        best_score = max([rf_acc, log_acc])
        directory_model = path + '/model.pkl'
        
        # save best model on disk
        if best_score == rf_acc:
            with open(directory_model, 'wb') as file:
                pickle.dump(rf, file)
                model = 'RandomForestClassifier'
                scoring_param = 'accuracy score'
                return (model, scoring_param, rf_acc)
        elif best_score == log_acc:
            with open(directory_model, 'wb') as file:
                pickle.dump(log_reg, file)
                model = 'LogisticRegression'
                scoring_param = 'accuracy score'
                return (model, scoring_param, log_acc)
    
    directory = path + '/train_data.pkl'
    with open(directory, 'rb') as file:
        train_data = pickle.load(file)
        
    # select features and target variable
    features = list(train_data)
    X = train_data[features]
    y = train_data[target]
    
    if model_type is None:
        if train_data[target].dtypes == np.object:
            feature_engineering = class_feature_selection_transformation(X, y, target, ordinal_feature, route, path)
            model, scoring_param, best_score = predict_class(feature_engineering, y, target, path)
            model_stats = [model, scoring_param, best_score, path]
            directory = path + '/model_stats.pkl'
            with open(directory, 'wb') as file:
                pickle.dump(model_stats, file)
                
        elif train_data[target].dtypes == np.float or train_data[target].dtypes == np.int:
            feature_engineering = regress_feature_selection_transformation(X, y, target, ordinal_feature, route, path)
            model, scoring_param, best_score = predict_regress(feature_engineering, y, target, path)
            model_stats = [model, scoring_param, best_score, path]
            directory = path + '/model_stats.pkl'
            with open(directory, 'wb') as file:
                pickle.dump(model_stats, file)
             
    elif model_type == 'classifier':
        feature_engineering = class_feature_selection_transformation(X, y, target, ordinal_feature, route, path)
        model, scoring_param, best_score = predict_class(feature_engineering, y, target, path)
        model_stats = [model, scoring_param, best_score, path]
        directory = path + '/model_stats.pkl'
        with open(directory, 'wb') as file:
            pickle.dump(model_stats, file)
            
    elif model_type == 'regressor':
        feature_engineering = regress_feature_selection_transformation(X, y, target, ordinal_feature, route, path)
        model, scoring_param, best_score = predict_regress(feature_engineering, y, target, path)
        model_stats = [model, scoring_param, best_score, path]
        directory = path + '/model_stats.pkl'
        with open(directory, 'wb') as file:
            pickle.dump(model_stats, file)
            

# app definition
app = flask.Flask(__name__)
app.config['DEBUG'] = True
UPLOAD_FOLDER = '/path/to/the/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response
    
@app.route('/train', methods=['POST'])
def train():
    route = request.path
    # retrieve data
    data = request.get_json(force = True)
    train_data = pd.DataFrame(data)
    
    # take arguments
    target = request.args['target']
    del_feature = request.args.get('del_feature')
    ordinal_feature = request.args.getlist('ordinal_feature')
    model_type = request.args.get('model_type')
    model_name = request.args['model_name']
    
    path = model_name
    if os.path.exists(path):
        return json.dumps({'Error': "Model already exists"})
    else:
        os.mkdir(path)
        directory = path + '/train_data.pkl'
        with open(directory, 'wb') as file:
            pickle.dump(train_data, file)
            
        download_thread = threading.Thread(target = build_model, args = (path, target, ordinal_feature, route, model_type,))
        download_thread.start()
        return json.dumps({'Message': 'Successful'})
    
        
    
@app.route('/predict', methods=['POST'])
def predict():
    route = request.path
    # retrieve data
    data = request.get_json(force = True)
    model_name = request.args['model_name']
    test_data = pd.DataFrame(data)

    path = model_name
    directory_fs = path + '/fs_values.pkl'
    directory_model = path + '/model.pkl'
    with open(directory_fs, 'rb') as file:
        unpickler = pickle.Unpickler(file);
        fs_values = unpickler.load();
        if len(fs_values) == 5:
            ordinal_feature, target = fs_values[3:] # regressor
        elif len(fs_values) == 4:
            ordinal_feature, target = fs_values[2:] # classifier
    with open(directory_model, 'rb') as file:
        model = pickle.load(file)

    feature_engineering = regress_feature_selection_transformation(test_data, None, None, ordinal_feature, route, path)
    y_predict = model.predict(feature_engineering)
    
    # merge 1st column with prediction
    df_1stcolumn = pd.DataFrame(test_data.iloc[:,0])
    df_prediction = pd.DataFrame({target: y_predict})
    df_1stcolumn = df_1stcolumn.reset_index(drop = True) # super important
    df_prediction = df_prediction.reset_index(drop = True) # super important
    output = df_1stcolumn.merge(df_prediction, left_index = True, right_index = True)
    result = output.to_json(orient = 'records')
    parsed = json.loads(result)
    
    return json.dumps(parsed, indent = 4)

@app.route('/status', methods=['GET'])
def status():
    model_name = request.args['model_name']
    directory = model_name + '/model_stats.pkl'
    if os.path.exists(directory):
        with open(directory, 'rb') as file:
            model, scoring_param, best_score, path = pickle.load(file)
            return json.dumps({'model': model, scoring_param :best_score, 'model name': path}, sort_keys = True, indent = 4, separators=(',', ': '))
    else:
        return json.dumps({'Message': 'Model is not available yet'})

@app.route('/delete', methods=['GET'])
def delete():
    target = request.args['target']
    if os.path.exists(target):
        shutil.rmtree(target, ignore_errors = True)
        return json.dumps({'Message': 'Directory successfully removed'})
    else:
        return json.dumps({'Error': 'Directory does not exist'})

if __name__ == "__main__":
    app.run(debug = True)
                                