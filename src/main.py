"""
title : Wireless terminal identification system
author : 김용환 (dragonbead95@naver.com)
date : 2020-10-26
detail : 
todo :

"""

import csv
import os
import prePro
import machine_learn
import file
import filePath
import copy
import numpy as np
import collect
import testset
import probe
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


"""probe-request 가공
probe-request를 전처리 및 학습 모델 생성
"""
def proReq_process():

    # step1 단말 분류
    data = probe.prepro_tag_length(filePath.learn_csv_probe_path) # tag length 필드 전처리

    data = probe.make_nullProReq(data)                            # null probe request 생성, length 필드 생성

    dfs = probe.separate_probe(data)                              # probe request를 단말별로 분류

    file.make_Directory(filePath.probe_path)                      # probe 디렉토리 생성

    dev_path ,devs = probe.save_separated_probe(dfs,filePath.probe_path)# 분류된 단말들을 csv 파일 형태로 each 저장

    
    for path, dev in zip(dev_path, devs): # path : ex) probe/dev1/
        # step2 단말 시퀀스 번호 전처리
        time_path = path+"time_separated/"
        file.make_Directory(time_path) # time_separated 디렉토리 생성
        time_names = probe.separate_time_probe(path,dev, time_path)

        # step3 feature 추출 및 저장
        featured_path = path+"featured/" # featured 디렉토리 생성
        file.make_Directory(featured_path)
        probe.write_feature(time_names,dev,featured_path)

        """step4 학습 데이터 생성
        
        """
        featured_csv = featured_path+"featured.csv"
        data = pd.read_csv(featured_csv)
        
        x_train = data[["delta_seq_no",
                    "wlan.ht.ampduparam.mpdudensity",
                    "wlan.ht.capabilities",
                    "wlan.ht.capabilities.rxstbc",
                    "wlan.ht.capabilities.txstbc",
                    "wlan.tag.length",
                    "length",
                    ]].values.tolist()

        target = data["label"].values.tolist()

        

        rf = RandomForestClassifier(n_estimators=100,random_state=0)
        rf.fit(x_train,target)
        
        

    # # 학습 데이터 생성
    # # feat_x_train : [[delta seq no, length], ..., [...]]
    # # feat_y_train : [0,0,1,1, ..., 2]
    # feat_x_train, feat_y_train = machine_learn.get_proReq_train_data(fm_name_list) 

    # # feature 통합 데이터 파일 생성
    # file.write_packet_test(feat_x_train,feat_y_train,filePath.packet_test_probe_path+"probe_train_test.csv")

    # # 무선단말 식별 모델 생성
    # device_model = machine_learn.random_forest_model(feat_x_train,feat_y_train)
    
    # # 무선단말 레이블 식별 모델 생성
    # device_label_model = machine_learn.label_random_forest(device_dic)

    # # 무선단말 식별 모델 저장
    # machine_learn.save_model(device_model,"device_model.pkl")

    # # 무선단말 식별 레이블 딕셔너리 저장
    # machine_learn.save_label_dic(device_dic,"device_label.json")
    
    # # 무선단말 레이블 식별 모델
    # machine_learn.save_model(device_label_model,"device_label_model.pkl")

    # # [ssid, wlan.sa], ex) "0" : ["carlynne","ff:ff:ff:ff:ff:ff"]
    # x_train, y_train, ap_dic = machine_learn.get_becon_train_data(bc_csv_fm_list) 
    
def main():
    while True:
        cmd_num = input("input the command\n"
                        +"1: init directory\n"
                        +"2: collect the packet\n"
                        +"3: filter the pcapng file\n"
                        +"4: training the ap/device\n"
                        +"5: delete the ap/device model\n"
                        +"6: create test set\n"
                        +"7: test the probe-request\n"
                        +"8: exit\n")

        if cmd_num=="1": # 디렉토리를 초기화한다.
            file.init_directory()

        elif cmd_num=="2": # 패킷 데이터를 수집한다.
            temp = input("input the network interface, duration, pcapname('wlan1' 3600 data.pcapng) : ").split(" ")
            neti, duration, pcapng_name = temp[0], temp[1], temp[2] # 네트워크인터페이스, 수집time, 저장파일이름
            collect.packet_collect(neti, duration, pcapng_name=pcapng_name)
            
        elif cmd_num=="3": # pcapng 파일을 필터링한다.
            # 필터링할 pcapng 파일 리스트를 출력한다.
            print(".pcapng file list")
            os.system("ls {} | grep '.*[.]pcapng'".format(filePath.pf_path))
            pcapng_name = filePath.pf_path +"/"+input("input the file name to filter the pcapng file(data.pcpapng) : ")

            collect.probe_filter(pcapng_name,filePath.learn_csv_probe_path)

        elif cmd_num=="4": # 학습용 데이터를 가공한다.
            proReq_process() # probe-request 가공

            # 무선 및 AP 식별 모델 불러오기
            if os.path.isfile("device_model.pkl"):
                device_model = machine_learn.load_model("device_model.pkl")
            if os.path.isfile("device_label.json"):
                device_dic = machine_learn.load_label_dic("device_label.json")
        elif cmd_num=="5":
            machine_learn.delete_model()

        elif cmd_num=="6": # 테스트용 데이터를 가공한다.
            # 필터링할 pcapng 파일 리스트 출력
            print(".pcapng file list")
            os.system("ls {} | grep '.*[.]pcapng'".format(filePath.pf_path))
            pcapng_name = filePath.pf_path +"/"+ input("input the file name to filter the pcapng file(data.pcpapng) : ")

            collect.probe_filter(pcapng_name,filePath.test_csv_probe_path)
            
            # Probe-request 테스트 데이터 가공 및 생성
            proReq_input = testset.proReq_createTestset() 
            
            # Probe-request 테스트 데이터 저장
            with open(filePath.packet_test_probe_csv_path,"w") as f:
                writer = csv.writer(f)
                writer.writerow(["length","delta seq no","label"])
                writer.writerows(proReq_input)

        elif cmd_num=="7":# 테스트용 데이터를 학습 모델에 넣어 식별결과를 확인한다.
            # 무선 식별 학습 모델 불러오기
            device_model = machine_learn.load_model("device_model.pkl")
            device_dic = machine_learn.load_label_dic("device_label.json")
            
            # probe-request 테스트 데이터 평가
            testset.packet_probe_test(device_model,device_dic)
            
        elif cmd_num=="8": # 프로그램 종료
            return;
        else:
            print("This is an invalid the command!!")

if __name__=="__main__":
    main()



