"""
title : 파일 작성 및 참조 관련 모듈
author : 김용환
date : 2020-11-29
"""

import os
import filePath

"""init_directory
detail : 프로젝트를 진행하는데 필요한 디렉토리를 초기 설정한다.
res
    - packet
        - probe
    - pcapng
    - pcapng_csv
        - learn
    - model
    - cmd_result
"""
def init_directory():
    make_Directory(filePath.res_path)               #res
    make_Directory(filePath.packet_path)            #packet
    make_Directory(filePath.probe_path)             #packet/probe
    make_Directory(filePath.pf_path)                #pcapng
    make_Directory(filePath.csv_path)               #pcapng_csv
    make_Directory(filePath.pcapng_csv_learn)       #pcapng_csv/learn
    make_Directory(filePath.model_path)             #model     
    make_Directory(filePath.cmd_result_path)        #cmd_result
 
"""make the Directory
경로상의 디렉토리를 생성한다.

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