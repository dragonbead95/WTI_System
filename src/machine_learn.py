"""
title : 기계학습 관련 모듈
author : 김용환
date : 2020-11-29
"""

import joblib
import filePath
import os

    
"""save the model
무선 단말 식별 모델을 파일로 저장한다.

param
model : 학습 식별 모델
filename : 모델을 저장할 경로 및 이름
"""
def save_model(model, filename):
    save_path = filePath.model_path + filename
    joblib.dump(model, save_path)

"""모델 불러오기
무선 단말 식별 모델 파일을 불러온다
param
filename : 파일 경로 및 이름
"""
def load_model(filename):
    load_path = filePath.model_path + filename
    if os.path.isfile(load_path):
        return joblib.load(load_path)
    else:
        return False

"""delete_model
detail : ap/device 식별 모델 삭제, model 디렉토리 안에 파일을 전부 삭제
"""
def delete_model():
    os.system("sudo rm -rf {}".format(filePath.model_path+"*"))

if __name__ == "__main__":
    pass
