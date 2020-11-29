"""
title : file path variables
author : 김용환
date : 2020-11-29

"""

import os 

path = os.getcwd()

res_path = path + "/../res/"

packet_path = res_path + "/packet/"
probe_path = packet_path + "/probe/"

pf_path = res_path + "/pcapng"
pf_data_path = pf_path + "/data.pcapng"

csv_path = res_path + "/pcapng_csv"
pcapng_csv_learn = csv_path + "/learn/"
learn_csv_probe_path = pcapng_csv_learn + "/probe.csv"

model_path = res_path + "/model/"
device_model_path = model_path + "/device_model.pkl"

cmd_result_path = res_path + "/cmd_result/"
feature_path_list_path = cmd_result_path + "feature_path_list.txt"
label_path = cmd_result_path + "labels.txt"

