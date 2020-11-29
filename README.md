# WTI_System
```
본 모듈은 무선 단말 식별 모듈입니다.
해당 모듈을 구현한 목적은 2개 이상의 AP로부터 수집되는 무선 단말들을 고유하게 식별하고자 목표로 합니다.
```
# 사용버전
```
python 3.8.2  
tensorflow 2.2.0  
sklearn 0.23.0  
```
# 실행 스크립트
sudo python3 main.py  

# 메뉴얼
```
1: init directory                   => 식별을 진행하기 위한 디렉토리 초기화입니다.

2: collect the packet               => 무선 랜 카드를 이용하여 패킷 데이터를 수집합니다.
                                       수집한 패킷 데이터는 pcapng 파일 형식으로 pcapng 디렉토리에 저장됩니다.
									   
3: filter the pcapng file           => pcapng 파일에서 probe request frame을 필터링하여
                                       csv형식으로 pcapng_csv/learn/probe.csv로 저장됩니다.
									   
4: training the ap/device           => probe request frame을 전처리 및 가공하여 기계학습 모델인 무선 단말 식별 모델에 학습시킵니다.

5: delete the device model / labels => 무선 단말 식별 모델 파일과 labels 파일을 제거합니다.

6: exit                             => 프로그램을 종료합니다.
```

```
step1 처음 수행시 1번 명령을 통하여 디렉토리 초기화  
step2 2번 명령을 통하여 패킷 데이터 수집 (pcapng 파일 이미 존재시 단계 생략 가능)  
step3 3번 명령 수행  
step4 4번 명령 수행  
step5 2개 이상의 ap에서 무선 단말 식별 진행을 원할 시 다른 pcapng 파일을 pcapng 디렉토리에 저장 후 step3~step4 과정 반복 수행  
```
# 폴더 구조
```
|  
|-res  
|  |  
|  |-cmd_result  
|  |  
|  |-model  
|  |  
|  |-packet  
|  |     |  
|  |     |-probe  
|  |          |  
|  |          |-dev1  
|  |          |-...  
|  |          |-devN  
|  |  
|  |-pcapng  
|  |  
|  |-pcapng_csv  
|        |  
|        |-learn  
|-src  
   |-collect.py  
   |-file.py  
   |-filePath.py  
   |-machine_learn.py  
   |-main.py  
   |-probe.py  
```
