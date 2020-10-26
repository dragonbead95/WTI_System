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
        dt.append(data["frame.time_relative"])
        ds.append(data["wlan.seq"])

        # step3-2 feature 2~7 추출을 위한 호출    
        mp.append(int(data["wlan.ht.ampduparam.mpdudensity"][0],16))
        cap.append(int(data["wlan.ht.capabilities"][0],16))
        cap_rx.append(int(data["wlan.ht.capabilities.rxstbc"][0],16))
        cap_tx.append(data["wlan.ht.capabilities.txstbc"][0])
        tag_leng.append(data["wlan.tag.length"][0])
        leng.append(data["length"][0])
            
            

    # step3-3 delta seq no 추출
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
    else:         # model 존재할때
        for i in range(len(new_data)):
            label = model.predict(new_data.iloc[i]) # 기존에 식별된 단말인지 식별
            label_probability = model.predict_proba(new_data.iloc[i]) # 예측 확률 판단
                        
            if label_probability>=0.6:
                labels.append(label)
        # 해당 부분 보완 필요 => json 파일을 통해서 추가하는 방법
        
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
            sum_tag_length = sum(list(map(int, line[8:])))  # tag length들의 합을 계산하여 저장
            dummy_line = line[:8]  # column 0~7 필드를 임시 저장
            dummy_line.append(sum_tag_length)  # 임시저장 리스트에 tag_length들의 합계 추가
            dummy.append(dummy_line)  # 새로 생성한 데이터 row 추가


    name = ["frame.time_relative","wlan.seq", "wlan.ssid", "frame.len", "wlan.ht.ampduparam.mpdudensity",
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
        filename = time_path + dev + "_" + str(i//6) + "_" + str((i%6)*10) + ".csv"
        ret.to_csv(filename, mode="w",index=False)

        time_separated_names.append(filename)

    return time_separated_names


"""
params
하나의 맥주소에 대해서 10분간격으로 분류된 csv파일을 참조하여 각각의 파일의
수신시관과 시퀀스넘버를 리스트 형태로 저장한다.

dev : mac주소
csvname : csv 파일 경로를 찾기 위한 문자열 경로

return
dt : 수신time 리스트
ds : 시퀀스 넘버 리스트

"""
def process_delta(dev,csvname="probe"):
    dev_name = []
    ap_name = []
    data_list = []
    data_size = []

    deltatime=[]
    deltaseq = []
    lost = []

    for i in range(144):
        dev_bssid = dev.replace(":","_")

        ospath = filePath.packet_path + "/" + csvname + "/" + dev_bssid
            
        filename = ospath + "/" + dev_bssid + "_" + str(i//6) + "_" + str((i%6)*10) + ".csv"

        try:
            df = pd.read_csv(filename)
            dev_name.append(filename)
            data_list.append(df)
            data_size.append(len(df))

            
            deltatime.append(df["timedifference"]) #해당 csv파일의 timedifference를 가져와 저장한다.
            deltaseq.append(df["sequence no"]) #해당 csv파일의 sequence no를 가져와 저장한다.
        except:
            lost.append([dev,i])
            continue
        
    
    dt = []
    ds = []
    for t,s in zip(deltatime, deltaseq):
        temp1 = []
        temp2 = []
        for i in range(len(t)):
            temp1.append(t[i]-t[0])
            temp2.append(s[i]-s[0])
        dt.append(temp1)
        ds.append(temp2)

    return dt, ds

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
            continue
        print(step, W_val, cost_val)
        pattern.append(W_val[0])
        pred.append(W_val*ds[i] + b_val)
        costt.append(tempcost)

    #delta seq no 평균을 구한다.
    print("Delta Seq No : {}".format(np.mean(pattern)))
    
    return pattern, np.mean(pattern)

if __name__ == "__main__":
    pass
