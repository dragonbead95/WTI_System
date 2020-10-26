"""
title : 파일 작성 및 참조 관련 모듈
author : 김용환 (dragonbead95@naver.com)
date : 2020-08-03
detail : 
todo :
"""

import os
import csv
import machine_learn
import filePath
import prePro
import pandas
import probe
import numpy as np
import copy
from scipy import stats

"""make directory
res
    model
    packet
        -beacon
        -beacon_test
        -probe
        -probe_test
    pcapng
    pcapng_csv
        -learn
        -test
    scan
        -beacon
        -probe
"""
def init_directory():
    make_Directory(filePath.res_path)               #res
    make_Directory(filePath.packet_path)            #packet
    make_Directory(filePath.probe_path)             #probe
    make_Directory(filePath.probe_test_path)        #probe_test
    make_Directory(filePath.pf_path)                #pcapng
    make_Directory(filePath.csv_path)               #pcapng_csv
    make_Directory(filePath.pcapng_csv_learn)       #learn
    make_Directory(filePath.pcapng_csv_test)        #test
    make_Directory(filePath.model_path)             #model
    make_Directory(filePath.scan_path)              #scan      
    make_Directory(filePath.scan_probe_path)        #probe
    make_Directory(filePath.packet_test)            #packet_test
    make_Directory(filePath.packet_test_probe_path) #probe

    
 
"""make the Directory
경로상의 디렉토리를 삭제하고 다시 경로상의 디렉토리를 생성한다.
찌꺼기 데이터가 남아있는것을 방지하기 위해서 한번 삭제한다.

params
path : 디렉터리를 생성할 경로
"""
def make_Directory(path):
    os.system("sudo rm -r "+path)

    if not os.path.exists(path):
        os.mkdir(path)
        os.system("sudo chmod 777 "
                        +path)      
        print("Directory ",path," created")
    else:
        print("Directory ",path," already exist")

"""mac주소 디렉터리 생성
mac_list에 있는 맥주소 이름의 디렉터리 생성

params
path : 맥주소 디렉터리가 저장되는 경로 중 일부
mac_list : 기기의 맥주소가 담격있는 맥주소 리스트
"""
def make_macDirectory(path,mac_list):
    for mac_name in mac_list:
        os.system("sudo rm -rf {}".format(path+mac_name))
        os.system("sudo mkdir {}".format(path+mac_name))

"""FeatureModel.csv 파일 생성
params
path : FeatureMOdel.csv 파일을 생성할 경로 중 일부
mac : 저장 파일 경로에 사용되는 맥주소
frame : seq -> probe-request의 경우 맥주소간의 콜론을 언더바(_)로 변경한다.
"""
def make_csvFeature(path,mac,frame="seq"):
    if frame=="seq":
        mac = mac.replace(":","_")

    csvFeatureFileName = path+mac+"/"+mac+"_"+"FeatureModel.csv"
    with open(csvFeatureFileName,"w") as f:
        writer = csv.writer(f)
        if frame=="seq":
            #writer.writerow(["delta seq no","length","label"])
            writer.writerow(["length","delta seq no","label"])

"""feature 데이터, label csv 파일을 생성하는 함수
데이터 필드 : length, delta seq no, label
"""
def write_packet_test(feat_x_train, feat_y_train,savename):
    x_train = copy.deepcopy(feat_x_train)
    y_train = copy.deepcopy(feat_y_train)
    
    for x, y in zip(x_train,y_train):
        x.extend(y)
    
    result = copy.deepcopy(x_train)

    with open(savename,"w") as f:
        writer = csv.writer(f)
        writer.writerow(["length","delta seq no","label"])
        writer.writerows(result)