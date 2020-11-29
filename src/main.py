"""
title : Wireless terminal identification system
author : 김용환
date : 2020-11-29
"""
import os
import machine_learn
import file
import filePath
import collect
import probe
import pandas as pd

def main():
    while True:
        cmd_num = input("input the command\n"
                        +"1: init directory\n"
                        +"2: collect the packet\n"
                        +"3: filter the pcapng file\n"
                        +"4: training the ap/device\n"
                        +"5: delete the device model / labels\n"
                        +"6: exit\n")

        if cmd_num=="1": # 디렉토리를 초기화한다.
            file.init_directory()

        elif cmd_num=="2": # 패킷 데이터를 수집한다.
            temp = input("input the network interface, duration, pcapname('wlan1' 3600 data.pcapng) : ").split(" ")
            neti, duration, pcapng_name = temp[0], temp[1], temp[2] # 네트워크인터페이스, 수집time, 저장파일이름
            collect.packet_collect(neti, duration, pcapng_name=pcapng_name)
            
            data = pd.read_csv(filePath.learn_csv_probe_path)
            data = data[data["wlan.sa"] in macs]
            data.to_csv(filePath.learn_csv_probe_path,index=False)

        elif cmd_num=="3": # pcapng 파일을 필터링한다.
            # 필터링할 pcapng 파일 리스트를 출력한다.
            print(".pcapng file list")
            os.system("ls {} | grep '.*[.]pcapng'".format(filePath.pf_path))
            pcapng_name = filePath.pf_path +"/"+input("input the file name to filter the pcapng file(data.pcpapng) : ")

            collect.probe_filter(pcapng_name,filePath.learn_csv_probe_path)

        elif cmd_num=="4": # 학습용 데이터를 가공한다.
            probe.proReq_process()
            data = probe.get_features()

            if os.path.isfile(filePath.device_model_path): # 식별모델이 존재하면 식별 수행
                probe.identify_ap(data)
            probe.train_model(data) # 식별 모델 학습
            
        elif cmd_num=="5": # 식별 모델, labels 전부 제거
            machine_learn.delete_model()
            os.system("sudo rm -rf {}".format(filePath.label_path))
        elif cmd_num=="6": # 프로그램 종료
            return;
        else:
            print("This is an invalid the command!!")

if __name__=="__main__":
    main()



