import pandas as pd
import os
import time
import filePath
import tensorflow.compat.v1 as tf
import collect
import prePro
import math
import numpy as np
import csv
import file
import machine_learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

"""label 저장
생성한 label을 csv파일에 저장한다.
dev : dev1, dev2, ...
"""
def save_label(dev,flag):
    num = dev.split("dev")
    num.remove('')
    num = num[0] # label number
    if flag==False:
        with open(filePath.label_path,"a") as f:
            f.write(num+" ")
        flag = True

    return flag
"""새로운 label 생성
csv파일을 참조하여 제일 큰 숫자에 +1을 하여 새로운 label 생성하고,
+1한 숫자는 다시 csv파일에 저장한다

return new_label (type : str)
"""
def create_label(flag):
    with open(filePath.label_path,"r") as f:
        line = f.readline() # ['1', ' ', '2', ' ']
        line = line.split(" ") # ['1','2', ..., '']
        line.remove('') # ['1','2']
        
        if flag==False:
            new_label = str(max(list(map(int,line)))+1)
        else:
            new_label = str(max(list(map(int,line))))
    
    return new_label

"""식별모델 학습
"""
def train_model(data):
    x_train = data[["delta_seq_no","wlan.ht.ampduparam.mpdudensity","wlan.ht.capabilities",
                    "wlan.ht.capabilities.rxstbc","wlan.ht.capabilities.txstbc","wlan.tag.length",
                    "length"]].values.tolist()

    target = data["label"].values.tolist()

    if os.path.isfile(filePath.device_model_path): # 기존 모델이 존재하면 추가 학습을 수행함
        rf = machine_learn.load_model("device_model.pkl")
    else: # 디바이스 모델이 존재하지 않으면 새로 생성
        rf = RandomForestClassifier(n_estimators=100,random_state=0) 
    
    rf.fit(x_train,target)

    machine_learn.save_model(rf,"device_model.pkl") # 식별모델 파일 저장

    

        


"""feature 가져오기
기계학습 모델에 넣기 위해 저장된 feature 들을 가져오는 기능
"""
def get_features():
    # 필요한 것 : 기기별로 저장된 featured.csv 경로
    os.system("sudo find {} -name featured.csv > \
                {}".format(filePath.probe_path, filePath.feature_path_list_path))

    # step1 featured.csv 경로가 저장된 txt파일을 참조하여 경로를 추출함
    with open(filePath.feature_path_list_path,"r") as f:
        feature_paths = f.read()
        feature_paths = feature_paths.split("\n") # 줄단위로 구분한 경로를 리스트화
        feature_paths.remove("") # '' 요소 제거
        
    # step2 featured.csv를 참조하여 feature value 추출
    datas = []
    for path in feature_paths:
        data = pd.read_csv(path)
        datas.append(
                data[["delta_seq_no","wlan.ht.ampduparam.mpdudensity","wlan.ht.capabilities",
                    "wlan.ht.capabilities.rxstbc","wlan.ht.capabilities.txstbc","wlan.tag.length",
                    "length","label"]]
                )
    #step3 feature 데이터 합치기
    return pd.concat(datas) # type:dataframe
        
    
        
    

"""ap식별
조건 : 식별 모델이 존재하면 모델에 넣어 식별 수행
"""
def identify_ap(data):
    
    # 식별 수행
    device_model = machine_learn.load_model("device_model.pkl")
    x_test = data[["delta_seq_no","wlan.ht.ampduparam.mpdudensity","wlan.ht.capabilities",
                    "wlan.ht.capabilities.rxstbc","wlan.ht.capabilities.txstbc","wlan.tag.length",
                    "length"]].values.tolist()

    y_tests = data["label"]

    predict_dev = device_model.predict(x_test) # ap 기기 식별 예측
    proba_dev = device_model.predict_proba(x_test)# 식별 예측 확률
    
    for pred, proba, y_test in zip(predict_dev,proba_dev,y_tests):
        if max(proba) > 0.6:
            print("pred : {}, proba : {}, real : {} is authorized".format(pred,max(proba),y_test))
        else:
            print("pred : {}, proba : {}, real : {} is unauthorized".format(pred,max(proba),y_test))
    print(pd.crosstab(y_tests,predict_dev))
    print(classification_report(y_tests,predict_dev))

"""probe-request 가공
probe-request를 전처리 및 학습 모델 생성
"""
def proReq_process():

    # step1 단말 분류
    data = prepro_tag_length(filePath.learn_csv_probe_path) # tag length 필드 전처리

    data = make_nullProReq(data)                            # null probe request 생성, length 필드 생성

    dfs = separate_probe(data)                              # probe request를 단말별로 분류

    file.make_Directory(filePath.probe_path)                      # probe 디렉토리 생성

    dev_path ,devs = save_separated_probe(dfs,filePath.probe_path)# 분류된 단말들을 csv 파일 형태로 each 저장

    for path, dev in zip(dev_path, devs): # path : ex) probe/dev1/
        # step2 단말 시퀀스 번호 전처리
        time_path = path+"time_separated/"
        file.make_Directory(time_path) # time_separated 디렉토리 생성
        time_names = separate_time_probe(path,dev, time_path)

        # timediffderence field 추가후 저장

        # step3 feature 추출 및 저장
        featured_path = path+"featured/" # featured 디렉토리 생성
        file.make_Directory(featured_path)
        write_feature(time_names,dev,featured_path)

"""단말에 대한 feature를 추출하고 저장한다
    feature 추출
    1~7 :   학습데이터
    8   :   정답 데이터

    1. delta seq no
    2. wlan.ht.ampduparam.mpdudensity
    3. wlan.ht.capabilities
    4. wlan.ht.capabilities.rxstbc
    5. wlan.ht.capabilities.txstbc
    6. wlan.tag.length
    7. length
    8. label

    delta seq no는 학습데이터 dt(frame.time_relative), 
    정답데이터 ds(wlan.seq)에 회귀분석한 기울기이다.
"""
def write_feature(time_names,dev,featured_path):   
    dt = [] # delta time
    ds = [] # delta seq
    mp = []
    cap = []
    cap_rx = []
    cap_tx = []
    tag_leng = []
    leng = []
    for filename in time_names:
        # step3-1 dt, ds 추출
        data = pd.read_csv(filename)
        dt.append(data["time_difference"])
        ds.append(data["seq_difference"])

        # step3-2 feature 2~7 추출을 위한 호출    
        mp.append(int(data["wlan.ht.ampduparam.mpdudensity"][0],16))
        cap.append(int(data["wlan.ht.capabilities"][0],16))
        cap_rx.append(int(data["wlan.ht.capabilities.rxstbc"][0],16))
        cap_tx.append(int(str(data["wlan.ht.capabilities.txstbc"][0]),16))
        tag_leng.append(data["wlan.tag.length"][0])
        leng.append(data["length"][0])
            
            
    # step3-3 delta seq no 추출
    #delta_seq_nos, delta_seq_no_avg = linear_regression(dt,ds)
    delta_seq_nos, delta_seq_no_avg = linear_regression(dt,ds)

    
    # step3-4 학습 dataframe 생성
    new_data = pd.DataFrame({"delta_seq_no":delta_seq_nos,
                        "wlan.ht.ampduparam.mpdudensity":mp,
                        "wlan.ht.capabilities":cap,
                        "wlan.ht.capabilities.rxstbc":cap_rx,
                        "wlan.ht.capabilities.txstbc":cap_tx,
                        "wlan.tag.length":tag_leng,
                        "length":leng
                        })

    # step3-5 label 생성
    labels = []
    model = machine_learn.load_model("device_model.pkl") 
    if not model: # model이 존재하지 않을때
        for idx in range(len(new_data)):
            labels.append(dev)
        save_label(dev,False) # 생성한 레이블 번호를 파일에 저장
        
    else:         # model 존재할때
        flag = False # 새로운 레이블 번호가 필요시, 계속된 레이블 번호 추가를 막기 위한 플래그 변수
        for i in range(len(new_data)):
            x_data = np.reshape(new_data.iloc[i].values.tolist(),(1,-1)) # 1d -> 2d
            
            label_pred = model.predict(x_data)[0] # 기존에 식별된 단말인지 식별
            label_proba = max(model.predict_proba(x_data)[0]) # 예측 확률 판단

            if label_proba>0.6:
                labels.append(label_pred)
            else: # 새로운 번호가 필요
                print("pred : {}, proba : {}".format(label_pred,label_proba))
                new_label= create_label(flag)
                labels.append("dev{}".format(new_label))
                flag = save_label("dev{}".format(new_label),flag)

    # step3-6 label 필드 추가
    new_data["label"] = labels

    # step3-7 featured.csv 작성
    new_data.to_csv(featured_path+"featured.csv",sep=',', na_rep='', index=False)

"""분류된 probe request 데이터들을 csv파일로 저장
1.dfs안에는 dataframe 타입의 each 분류된 단말 probe request 데이터 리스트로 저장됨
2. csv 파일들은 1~N까지 1.csv ..., N.csv 형태로 저장

param
dfs : dataframe 타입의 분류된 probe request 데이터들


return
paths : [probe/dev1/, probe/dev2/, ..., probe/devn/]

"""
def save_separated_probe(dfs,savepath):
    filename = 1
    paths = []
    devs = []
    for df in dfs:
        if len(df) >= 100:  # probe request data가 100개 이상 수집된 단말만 분류 csv 파일 생성
            path = savepath+"dev{}/".format(filename) # 저장 디렉토리
            dev_name = "dev{}".format(filename)   # dev1.csv
            
            devs.append(dev_name)                     # [dev1.csv, dev2.csv, ..., devn.csv]
            paths.append(path)                        # dev 저장 디렉토리 경로 저장
            
            file.make_Directory(path)                 # 단말 디렉토리 생성
            df.to_csv(path+dev_name+".csv", sep=',', na_rep='', index=False) # csv 파일 생성
            filename += 1

    return paths, devs

""" probe request 데이터모음을 단말별로 분류할 기준 생성
분류 기준
wlan.ht.ampduparam.mpdudensity
wlan.ht.capabilities
wlan.ht.capabilities.rxstbc
wlan.ht.capabilities.txstbc
wlan.tag.length
length

stand = [('0x00000000', '0x0000016e', '0x00000001', '0', 46, 106), ...] (식별자 묶음)

return
dfs : 분류 dataframe 타입 단말들의 리스트
"""
def separate_probe(data):
    # 분류기준에 따른 그룹생성 -> 단말들을 분류할 기준 생성
    grouped = data.groupby([
        data["wlan.ht.ampduparam.mpdudensity"],
        data["wlan.ht.capabilities"],
        data["wlan.ht.capabilities.rxstbc"],
        data["wlan.ht.capabilities.txstbc"],
        data["wlan.tag.length"],
        data["length"]
    ])

    stand = []  # 단말을 분류 식별자 묶음들
    for name, group in grouped:
        stand.append(name)


    # 분류 기준 딕셔너리
    filter_obj = {
        "wlan.ht.ampduparam.mpdudensity": "wlan.ht.ampduparam.mpdudensity",
        "wlan.ht.capabilities": "wlan.ht.capabilities",
        "wlan.ht.capabilities.rxstbc": "wlan.ht.capabilities.rxstbc",
        "wlan.ht.capabilities.txstbc": "wlan.ht.capabilities.txstbc",
        "wlan.tag.length": "wlan.tag.length",
        "length": "length"
    }

    # 분류 기준에 따른 단말 분류 수행
    dfs = []
    for item in stand:
        dfs.append(data[(data[filter_obj["wlan.ht.ampduparam.mpdudensity"]] == item[0]) &
                        (data[filter_obj["wlan.ht.capabilities"]] == item[1]) &
                        (data[filter_obj["wlan.ht.capabilities.rxstbc"]] == item[2]) &
                        (data[filter_obj["wlan.ht.capabilities.txstbc"]] == item[3]) &
                        (data[filter_obj["wlan.tag.length"]] == item[4]) &
                        (data[filter_obj["length"]] == item[5])])

    return dfs

""" null probe request 필드 생성
length = frame.len - wlan.ssid의 길이
dataframe에 length 필드 추가
"""
def make_nullProReq(data):
    leng = []
    # null probe request 가공
    for idx in range(len(data)):
        leng.append(int(data.iloc[idx]["frame.len"]) -
                    len(data.iloc[idx]["wlan.ssid"]))
    leng_df = pd.DataFrame({"length": leng})
    data["length"] = leng_df  # length 필드 생성
    return data

""" tag_length 합계 계산
tag length 필드 필터링 시 -> 0, 8, 4, 8, ..., 16
위와 동일한 형태로 csv파일에 저장
문제는 seperator="," 이기 때문에 필드 열이 맞지가 않고
지문으로 사용할 tag_length들의 합계를 전처리하여 새로운 dataframe 타입 생성 필요

기존
frame.time_relative,wlan.seq,wlan.ssid,frame.len,wlan.ht.ampduparam.mpdudensity,wlan.ht.capabilities,wlan.ht.capabilities.rxstbc,wlan.ht.capabilities.txstbc,wlan.tag.length
4.261908020,422,,159,0x00000005,0x0000002d,0x00000000,0,0,4,8,1,26,8,1,5,19,8,9

변경
frame.time_relative,wlan.seq,wlan.ssid,frame.len,wlan.ht.ampduparam.mpdudensity,wlan.ht.capabilities,wlan.ht.capabilities.rxstbc,wlan.ht.capabilities.txstbc,wlan.tag.length
4.261908020,422,,159,0x00000005,0x0000002d,0x00000000,89

return data (type:dataframe,변경된 형식과 동일한 형태로 반환)
"""
def prepro_tag_length(filename):
    # tag_length들의 합을 계산한다.
    dummy = []
    with open(filename, "r", encoding="UTF-8") as f:
        rdr = csv.reader(f)
        next(rdr)
        for line in rdr:
            sum_tag_length = sum(list(map(lambda x: x if type(x)==int else int(x,16),line[9:])))  # tag length들의 합을 계산하여 저장
            dummy_line = line[:9]  # column 0~8 필드를 임시 저장
            dummy_line.append(sum_tag_length)  # 임시저장 리스트에 tag_length들의 합계 추가
            dummy.append(dummy_line)  # 새로 생성한 데이터 row 추가


    name = ["wlan.sa","frame.time_relative","wlan.seq", "wlan.ssid", "frame.len", "wlan.ht.ampduparam.mpdudensity",
            "wlan.ht.capabilities", "wlan.ht.capabilities.rxstbc", "wlan.ht.capabilities.txstbc",
            "wlan.tag.length"]
    data = pd.DataFrame(dummy, columns=name)  # convert list to dataframe
    
    return data

"""probe.csv들을 하나로 합침

params
filename : .csv 파일 경로

return
data : DataFrame type
"""
def read_probe(file_list):
    df_list = []    # dataFrame list

    # probe.csv 파일들을 리스트에 추가
    for file in file_list:
        df = pd.read_csv(file, error_bad_lines=False).fillna("")
        df_list.append(df)
        
    # probe.csv 데이터들을 하나로 합쳐서 반환
    return pd.concat(df_list)

    
    
"""시퀀스 번호 전처리 및 time별 분류저장
params
path : packet/probe/dev1/
dev : dev1.csv
time_path : patcket/probe/dev1/time_separated

detail:
wlan.seq : 0~4096 cycle -> 0~infinite 전처리 수행 및 difference 형태로 저장
frame.time_relative : difference 형태로 저장

10분 gap으로 csv파일 분할 저장

"""
def separate_time_probe(path, dev, time_path):
    
    time_separated_names = [] # [dev1_0_0.csv, ...]

    dev_csv = path+dev+".csv"
    
    data = pd.read_csv(dev_csv).fillna("")

    indd=[]     # circle 변경 인덱스
    timedif=[] 
    seqno= []   
    for i in range(len(data)):
        #시퀀스 넘버의 사이클이 변경되는 지점(index)를 indd 리스트에 저장
        if i != 0 and data.iloc[i]["wlan.seq"] - data.iloc[i-1]["wlan.seq"] < 0:
            indd.append(i)
                    
        #time을 index 0부터 시작하기 위해서 0번째 time을 빼주어 싱크로를 맞춘다.
        timedif.append(data.iloc[i]["frame.time_relative"] - data.iloc[0]["frame.time_relative"])


    #시퀀스 넘버 전처리
    for i in range(len(indd)):
        if i == len(indd)-1:    #i가 마지막 인덱스인 경우
            data.iloc[indd[i]:]["wlan.seq"] = data.iloc[indd[i]:]["wlan.seq"] + 4096 *(i+1)
        else:
            data.iloc[indd[i]:indd[i+1]]["wlan.seq"] = data.iloc[indd[i]:indd[i+1]]["wlan.seq"] + 4096 * (i+1)

    # get wlan.seq difference
    for i in range(len(data)):
        seqno.append(data.iloc[i]["wlan.seq"] - data.iloc[0]["wlan.seq"])

    # column data update
    data.update(pd.DataFrame({"frame.time_relative":timedif}))
    data.update(pd.DataFrame({"wlan.seq":seqno}))


    #csv를 10분 gap으로 분류하여 저장한다.
    for i in range(144):
        ret = data[data["frame.time_relative"] >= (i*600)][data["frame.time_relative"]<600 * (i+1)]
        if len(ret) < 14:
            continue

        ret["time_difference"] = ret["frame.time_relative"]-ret.iloc[0]["frame.time_relative"] # time_difference field 추가
        ret["seq_difference"] = ret["wlan.seq"] - ret.iloc[0]["wlan.seq"] # seq_difference field 추가


        filename = time_path + dev + "_" + str(i//6) + "_" + str((i%6)*10) + ".csv"
        ret.to_csv(filename, mode="w",index=False)

        time_separated_names.append(filename)

    return time_separated_names

def linear_regression(dt, ds):
    tf.disable_v2_behavior()

    W = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])

    hypothesis = X * W + b

    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000000005)
    train = optimizer.minimize(cost)
    pattern = []
    pred = []
    costt = []
    sess = tf.Session()

    for i in range(len(dt)):
        sess.run(tf.global_variables_initializer())
        tempcost = []
        
        for step in range(501):
            _, cost_val, W_val, b_val = sess.run([train, cost, W, b],feed_dict={X:dt[i], Y:ds[i]})
            tempcost.append(W_val)
    
        if math.isnan(W_val[0]):
            print("raise nan")
            continue
        print(step, W_val, cost_val)
        pattern.append(W_val[0])
        pred.append(W_val*ds[i] + b_val)
        costt.append(tempcost)

    #delta seq no 평균을 구한다.
    print("Delta Seq No : {}".format(np.mean(pattern)))
    
    return pattern, np.mean(pattern)



if __name__ == "__main__":
    save_label("dev10")
