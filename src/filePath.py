"""
title : file path variables
author : YONG HWAN KIM (yh.kim951107@gmail.com)
date : 2020-07-13
detail : 
todo :
"""

import os 

path = os.getcwd()

res_path = path + "/../res/"

packet_path = res_path + "/packet/"
probe_path = packet_path + "/probe/"
probe_test_path = packet_path + "/probe_test/"

pf_path = res_path + "/pcapng"
pf_data_path = pf_path + "/data.pcapng"

csv_path = res_path + "/pcapng_csv"
pcapng_csv_learn = csv_path + "/learn/"
pcapng_csv_test = csv_path + "/test/"
learn_csv_probe_path = pcapng_csv_learn + "/probe.csv"
learn_csv_probeRe_path = pcapng_csv_learn + "/probe_re.csv"
test_csv_probe_path = pcapng_csv_test + "/probe.csv"
test_csv_probeRe_path = pcapng_csv_test + "/probe_re.csv"

model_path = res_path + "/model/"
device_model_path = model_path + "/device_model.pkl"
device_label_path = model_path + "/device_label.json"
device_label_model_path = model_path + "/device_label_model.pkl"

scan_path = res_path + "/scan/"
scan_probe_path = scan_path + "/probe/"

packet_test = res_path + "/packet_test/"
packet_test_probe_path = packet_test + "/probe/"
packet_test_probe_csv_path = packet_test_probe_path + "/probe_test.csv"

cmd_result_path = res_path + "/cmd_result/"
feature_path_list_path = cmd_result_path + "feature_path_list.txt"
label_path = cmd_result_path + "labels.txt"

