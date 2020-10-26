"""
title : mechine learing moudle
author : YONG HWAN KIM (yh.kim951107@gmail.com)
date : 2020-07-15
detail : 
todo :
"""

import csv
import numpy as np
import prePro
import pandas as pd
import tensorflow.compat.v1 as tf
import pickle
import joblib
import filePath
import json
import test
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN

"""clock skew 기울기 계산

becon-frame
x_train : wlan.fixed.timestamp list
y_train : i th frame.time_relative - 0th frame.time_relative) - i th wlan.fixed.timestamp, data list

return
line_fitter.coef_ : clock skew
"""
def sklearn_linear_regression(x_train,y_train):
    
    X = np.array(x_train).astype(np.float64).reshape(-1,1)
    y = np.array(y_train).astype(np.float64)
    
    line_fitter = LinearRegression()
    line_fitter.fit(X,y)
    
    return line_fitter.coef_

##############################################################################
"""특징 데이터를 가져온다.
params
name : featuremodel.csv 파일 이름 경로

return
x_train : [[delta seq no, length], ... , [...]]
y_train : [[label], ... , [...]]
"""
def get_proReq_FeatureModel(name):
    x_train = []
    y_train = []
    print("name : {}".format(name)) #debug
    print("name type : {}".format(type(name))) #debug
    with open(name,"r") as f:
        rdr = csv.reader(f)
        next(rdr,None) #header skip
        
        for row in rdr:
            x_train.append(list(map(float,row[:2])))  # [length, delta seq no]
            """
            if row[-1]=="-1":                         # -1인 경우 매핑하지 않고 문자열 형식으로 저장한다.
                y_train.append([row[-1]])
            else:
                y_train.append(list(map(int,row[-1])))  # [label]
            """
            y_train.append(row[-1])

    return x_train, y_train

"""probe-request 학습데이터를 가져온다.
featuremodel.csv 파일들을 순회하며 학습데이터를 리스트에 저장한다.

parmas
fm_name_list : featuremodel.csv 파일 경로 리스트

return
feat_x_train : [[sequence number delta, length], ..., [...]]
feat_y_train : [label, ...]
"""
def get_proReq_train_data(fm_name_list):
    feat_x_train = []
    feat_y_train = []
    
    #featuremodel.csv 파일 리스트중에서 하나를 가져와 해당 featuremodel.csv 파일의 학습 데이터를 가공한다.
    for name in fm_name_list:
        x_train, y_train = get_proReq_FeatureModel(name)
        
        for data in x_train: #x_train 리스트를 feat_x_train 리스트에 누적한다.
            feat_x_train.append(data)
        for data in y_train: #y_train 리스트를 feat_y_train 리스트에 누적한다.
            feat_y_train.append(data)

    return feat_x_train, feat_y_train

"""학습 식별 모델을 생성한다.
학습 모델 타입은 랜덤 포레스트이다.
"""
def random_forest_model(x_train, target):

    if os.path.isfile(filePath.device_model_path): # 기존 모델이 존재하면 추가 학습을 수행함
        rf = load_model("device_model.pkl")
    else: # 디바이스 모델이 존재하지 않으면 새로 생성
        rf = RandomForestClassifier(n_estimators=100,random_state=0) 
    rf.fit(x_train,target)

    return rf
    
"""save the model
param
model : 학습 식별 모델
filename : 모델을 저장할 경로 및 이름
"""
def save_model(model, filename):
    save_path = filePath.model_path + filename
    joblib.dump(model, save_path)

"""save the label dictionary
param
dic : 레이블 딕셔너리
filename : 딕셔너리를 저장할 경로 및 이름
"""
def save_label_dic(dic,filename):
    save_path = filePath.model_path + filename
    
    with open(save_path,"w") as json_file:
        json.dump(dic,json_file)

"""모델 불러오기
param
filename : 파일 경로 및 이름
"""
def load_model(filename):
    load_path = filePath.model_path + filename
    if os.path.isfile(load_path):
        return joblib.load(load_path)
    else:
        return False

"""레이블 딕셔너리 불러오기
param
filename : 파일 경로 및 이름
"""
def load_label_dic(filename):
    load_path = filePath.model_path + filename

    with open(load_path,"r") as json_file:
        dic = json.load(json_file)
    return dic

"""ap/device 식별 모델 삭제
ap/device 식별 모델 및 label 식별모델, json 파일을 삭제한다.
"""
def delete_model():
    os.system("sudo rm -rf {}".format(filePath.model_path+"*"))

def label_random_forest(device_dic):

    data = []
    target = []
    for key, feature in device_dic.items():
        #data.append([float(feature[0]),int(feature[1])]) # [delta seq no, length]
        data.append([int(feature[0]),float(feature[1])]) # [delta seq no, length]
        target.append(int(key))

    rf = RandomForestClassifier(n_estimators=100,random_state=0)
    rf.fit(data,target)

    return rf

if __name__ == "__main__":
    pass
