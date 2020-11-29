"""
title : 패킷 수집 및 필터링 관련 모듈
author : 김용환
date : 2020-11-29
detail : 
todo :
"""

import os
import filePath
import csv
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
filename :    저장할 csv 파일
"""
def probe_filter(pcapng_name, filename):
    os.system("sudo tshark -r "
                    + pcapng_name
                    + " -Y \"wlan.fc.type_subtype==0x0004\""
                    + " -T fields"
                    + " -e frame.time_relative"
                    + " -e wlan.seq"
                    + " -e wlan.ssid"
                    + " -e frame.len"
                    + " -e wlan.ht.ampduparam.mpdudensity"
                    + " -e wlan.ht.capabilities"
                    + " -e wlan.ht.capabilities.rxstbc"
                    + " -e wlan.ht.capabilities.txstbc"
                    + " -E separator=, -E quote=n -E header=y > "
                    + filename)

if __name__ == "__main__":
    pass