"""
title : preprocessor about frame
author : YONG HWAN KIM (yh.kim951107@gmail.com)
date : 2020-08-03
detail : 
todo :
"""

import pandas as pd
import csv
import copy

"""맥주소 추출
params
filename : 파일 이름

return
mac_list : 중복되지 않는 맥주소 리스트
"""
def extract_macAddress(filename):
    data = pd.read_csv(filename)
    mac_list = list(set(data["wlan.sa"])) # set을 통하여 중복없는 맥주소 집합 생성 후 리스트화
    mac_list.sort()
    return mac_list


"""패킷 데이터 추출
params
path : 추출하고자 하는 파일
mac_list : wlan.sa (맥주소) 리스트

return
mac_dc : key:wlan.sa value: packet
"""
def extract_packetLine(path,mac_list):
    mac_dc = {}

    for mac_name in mac_list:   # dictionary 초기화
        mac_dc.update({mac_name:[]})
    
    with open(path,"r") as f:
        rdr = csv.reader(f)
        packet_list=[]

        #데이터 추출
        for line in rdr:
            packet_list.append(line);

        #맥별 패킷 분류
        for mac_name in mac_list: 
            for line in packet_list:
                if line[0]==mac_name:
                    mac_dc[mac_name].append(line)
    
    return mac_dc

"""시간변환
sec와 interval을 입력으로 얻어서 시,분으로 변환한다.

params
sec : second
interval : probe-request => 10, becon-frame => 3

return
hour, minute
"""
def trans_time(sec,interval):
    s = sec

    h = int(s // 3600)
    s = int(s%3600)

    m = int(s//60)
    m = (m//interval)*interval
    
    if h<10:
        h = "0"+str(h)
    if m<10:
        m = "0"+str(m)

    return str(h), str(m)
