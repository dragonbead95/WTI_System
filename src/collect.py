"""
title : 패킷 수집 및 필터링 관련 모듈
author : 김용환 (dragonbead95@naver.com)
date : 2020-09-07
detail : 
todo :
"""

import os
import filePath
import csv
import dev
import copy
import pandas as pd

"""패킷 데이터 수집
params
neti : 네트워크 인터페이스
sec : 수집time(초/s)
pcapng_name : 저장할 pcapng 파일 이름
"""
def packet_collect(neti, sec,pcapng_name="data.pcapng"):
    # 랜카드를 모니터 모드로 설정
    os.system("sudo ifconfig {} down".format(neti))
    os.system("sudo iwconfig {} mode monitor".format(neti))
    os.system("sudo ifconfig {} up".format(neti))
     
    # 패킷 수집 명령어
    os.system("sudo tshark -i {} -w ".format(neti)
                    + filePath.pf_path
                    + "/" + pcapng_name
                    + " -f \'wlan type mgt and (subtype beacon or subtype probe-req)\'"
                    + " -a duration:{}".format(sec))

"""pcap 파일 probe request 필터링
pcap 파일을 probe request 데이터로 필터링후 csv 파일 생성
params
pcapng_name : 필터링할 pcap 파일
filename : 저장할 csv 파일
"""
def probe_filter(pcapng_name, filename):
    os.system("sudo tshark -r "
                    + pcapng_name
                    + " -Y \"wlan.fc.type_subtype==0x0004\""
                    + " -T fields"
                    + " -e wlan.sa"
                    + " -e frame.time_relative"
                    + " -e wlan.seq"
                    + " -e wlan.ssid"
                    + " -e frame.len"
                    + " -e wlan.ht.ampduparam.mpdudensity"
                    + " -e wlan.ht.capabilities"
                    + " -e wlan.ht.capabilities.rxstbc"
                    + " -e wlan.ht.capabilities.txstbc"
                    + " -e wlan.tag.length"
                    + " -E separator=, -E quote=n -E header=y > "
                    + filename)

"""
무선단말을 필터링하여 probe.csv 파일에 재저장
params
filename : 필터링할 csv 파일 이름
"""
def probe_filter_testcase(filename):

    dev_dic = dev.dev_dic                               # 무선단말 기기
    dev_list = list(dev_dic.values())                   

    data = pd.read_csv(filename)                        # 필터링할 csv 파일 읽기
    data = data[data["wlan.sa"].isin(dev_list)]         # dev_list안에 있는 맥주소로 필터링

    data.to_csv(filename,sep=",",na_rep="",index=False) # 필터링한 csv 파일 작성

if __name__ == "__main__":
    pass