import os
import subprocess
import statistics
import json
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
from datetime import datetime
import sys

mac_to_device = {}
mac_to_ip = {}
with open("devices.txt", "r") as devices_list:
    for line in devices_list.readlines():
        line = line.strip().split(",")
        mac_to_device[line[0].strip()] = (line[1].strip())
        #mac_to_ip[line[1].strip()] = (line[2].strip())

# separate pcap file per each one hour window
hourly_pcaps_dir = "pcaps/hourly/"

all_ports_ever = set()
all_domains_ever = set()
all_cs_ever = set()

#ports = []
#domains = []
#css = []

records = []]
instances = {}
#for mac in mac_to_device.keys():
for mac in [sys.argv[1]]:
    instances[mac] = []
    print(f"Processing {mac}")
    if not os.path.exists(hourly_pcaps_dir + mac):
        print(hourly_pcaps_dir + mac + "does not exists")
        continue
    for hourly_instnace in os.listdir(hourly_pcaps_dir + mac):
        time_start = 0
        remote_ports = []
        dns_names = []
        cs = []

        flow_volumes = []
        flow_durations = []
        dns_times = []

        sleep_time = 3600
        latest_end = 0
        
        # joy tool from Cisco needs to be installed
        task = subprocess.Popen("joy bidir=1 tls=1 dns=1 " + hourly_pcaps_dir + mac + "/" + hourly_instnace, shell=True, stdout=subprocess.PIPE)
        data = task.stdout.read().decode()
        #assert task.wait() == 0
        if task.wait() != 0:
            print(f"Error processing {mac}/{hourly_instnace}")
        for line in data.split("\n"):
            if not len(line) > 0:
                continue
            json_obj = json.loads(line)
            if "sa" in json_obj:
                if time_start == 0:
                    time_start = json_obj["time_start"]
                # Add remote port
                if "192.168" in json_obj["sa"] and "192.168" not in json_obj["da"]:
                    if type(json_obj["dp"]) == type(0):
                        remote_ports.append(str(json_obj["dp"]))
                        all_ports_ever.add(str(json_obj["dp"]))

                # Calculate sleep time for the 3600 seconds over which pcap file is generated
                if json_obj["time_start"] >= latest_end:
                    sleep_time -= json_obj["time_end"] - json_obj["time_start"]
                    latest_end = json_obj["time_end"]
                else:
                    if json_obj["time_end"] <= latest_end:
                        pass
                    else:
                        sleep_time -= json_obj["time_end"] - latest_end
                        latest_end = json_obj["time_end"]

                # Add DNS name
                if "dns" in json_obj:
                    for item in json_obj["dns"]:
                        try:
                            dns_names.append(item["qn"])
                            all_domains_ever.add(item["qn"])
                        except:
                            dns_names.append(item["rn"])
                            all_domains_ever.add(item["rn"])
                    dns_times.append(json_obj["time_start"])

                # Add CS
                if "tls" in json_obj:
                    if "cs" in json_obj["tls"]:
                        cs.append(''.join(json_obj["tls"]["cs"]))
                        all_cs_ever.add(''.join(json_obj["tls"]["cs"]))

                # Add flow information if a biflow
                if "bytes_in" in json_obj and "bytes_out" in json_obj:
                    flow_volumes.append(json_obj["bytes_in"] + json_obj["bytes_out"])
                    flow_durations.append(json_obj["time_end"] - json_obj["time_start"])

        if len(dns_times) > 2:
            temp = []
            for i in range(1, len(dns_times)):
                temp.append(dns_times[i] - dns_times[i-1])
            dns_interval = statistics.mean(temp)
        else:
            dns_interval = 0

        # print(mac, remote_ports)

        # Ensure we have read at least one flow from the pcap
        if len(flow_volumes) >= 1 and statistics.mean(flow_durations) > 0:
            instances[mac].append([remote_ports, dns_names, cs, statistics.mean(flow_volumes),
                               statistics.mean(flow_durations), statistics.mean(flow_volumes) / statistics.mean(flow_durations),
                               sleep_time, dns_interval])
            record = {'filename': hourly_instnace, 'time_start': datetime.fromtimestamp(time_start), 'mac': mac, 'ports': dict(Counter(remote_ports)), 'dns': dict(Counter(dns_names)), 'cs': dict(Counter(cs)), 
                      'volume_mean': statistics.mean(flow_volumes),
                      'flow_durations': statistics.mean(flow_durations), 'ratio': statistics.mean(flow_volumes) / statistics.mean(flow_durations),
                      'sleep_time': sleep_time, 'dns_interval': dns_interval}
            records.append(record)
        else:
            print("NOTIFY: possibly zero flows in the file ", hourly_instnace, " for ", mac)

df = pd.DataFrame(records)
df.to_csv(f'unsw_features_{mac}.csv', index=False)

sys.exit()

