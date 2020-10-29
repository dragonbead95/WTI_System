import pandas as pd
import csv
import filePath
macs = ["84:2e:27:6b:53:df","00:f4:6f:9e:c6:eb","94:d7:71:fc:67:c9","18:83:31:9b:75:ad"]
temp = []
with open(filePath.learn_csv_probe_path,"r",encoding="UTF-8") as f:
    rdr = csv.reader(f)
    next(rdr)
    for line in rdr:
        if line[0] in macs:
            temp.append(line)


name = ["wlan.sa","frame.time_relative","wlan.seq", "wlan.ssid", "frame.len", "wlan.ht.ampduparam.mpdudensity",
            "wlan.ht.capabilities", "wlan.ht.capabilities.rxstbc", "wlan.ht.capabilities.txstbc",
            "wlan.tag.length"]
data = pd.DataFrame(line,columns=name)
data.to_csv(filePath.learn_csv_probe_path, index=False)