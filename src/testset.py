"""
title : Wireless terminal identification system
author : YONG HWAN KIM (yh.kim951107@gmail.com)
date : 2020-07-15
detail : 
todo :
"""

import file
import filePath
import machine_learn
import prePro
import numpy as np
import collect
import probe
import copy
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

"""무선 단말 테스트 데이터 생성

return
feat_x_train : [[delta seq no, length, label], ...]
"""
def proReq_createTestset():
    mac_list = []       # wlan.sa list, list["fe:e6:1a:f1:d6:49", ... ,"f4:42:8f:56:be:89"]
    feat_x_train = []   # random forest model x_train
    feat_y_train = []   # random forest model y_train
    device_dic = {}     # key:label value: mac address
    label = 0

    collect.probe_filter_testcase(filePath.test_csv_probe_path)

    mac_list = prePro.extract_macAddress(filePath.test_csv_probe_path)   # 맥주소 추출

    data = probe.read_probe(filePath.test_csv_probe_path)

    file.make_Directory(filePath.probe_test_path)

    probe.separate_probe2(mac_list,data,csvname="probe_test")

    # make feature csv file for each the wlan.sa
    for mac_name in mac_list:
        file.make_csvFeature(filePath.probe_test_path,mac_name,"seq")

    device_dic = machine_learn.load_label_dic("device_label.json")
    
    fm_name_list, device_dic = file.init_seq_FeatureFile(data,filePath.probe_test_path,device_dic=device_dic,csvname="probe_test",mode="test") #a dd the feature data

    print("fm_name_list : {}".format(fm_name_list))
    feat_x_train, feat_y_train = machine_learn.get_proReq_train_data(fm_name_list) # 학습 데이터 생성

    
    feat_y_train = np.reshape(feat_y_train,(-1,1)) # [0,1,2] => [[0],[1],[2]]

    #[[delta seq no, length, label]...,] 형식으로 생성
    for x, y in zip(feat_x_train, feat_y_train):
        x.extend(y)
    
    return feat_x_train

"""테스트 데이터 평가
params
model : 랜덤 포레스트 학습 모델
dic : probe-request -> key : label, value : wlan.sa
      becon-frame -> key : label, value : (SSID, wlan.sa)

x_input : probe-request -> [[delta seq no, length], ...]
          becon-frame -> [[[clock skew, channel, rss, duration, ssid, mac address], ...]
y_test : probe-request -> [label]
         becon-frame -> [label]
"""
def packet_probe_test(model, dic):

    data = pd.read_csv(filePath.packet_test_probe_csv_path)
    labels = pd.DataFrame(data["label"])
    feature = data[["length","delta seq no"]]

    predict = pd.DataFrame(model.predict(feature))
    predict.columns = ["predict"]

    r = pd.concat([feature,labels,predict], axis=1)

    
    ct = pd.crosstab(data["label"],r["predict"])
    print("device identify result")
    print("data length : {}".format(len(data)))
    print(ct)
    print()

    # debug
    # for index, row in data.iterrows():
    #     if row["label"]== (-1):
    #         print("length: {}, delta seq no: {}, result: {}".format(row["length"],row["delta seq no"],"unknown device"))
    #     else:
            
    #         pred = model.predict([row[["length","delta seq no"]]])[0]
    #         proba = max(model.predict_proba([row[["length","delta seq no"]]])[0])
            
    #         if proba > 0.6:
    #             print("length: {}, delta seq no: {}, result: {}".format(row["length"],row["delta seq no"],"authorized device"))
    #         else:
    #             print("length: {}, delta seq no: {}, result: {}".format(row["length"],row["delta seq no"],"mac spoofing device"))
    
    
    

