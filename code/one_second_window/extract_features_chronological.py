import pandas as pd
import os
import sys

'''
The network traffic made available by the authors of 'Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics' is composed of IoT and non-IoT devices.
 
We implemented this Python script to extract IoT devices from datasets and assign labels to them based on their mac addresses. 
'''

#The folder containing the datasets.
#folder = str(sys.argv[1])

#Maps the MAC address of an IoT device to a label (integer).
#devices ={"d0:52:a8:00:67:5e":1,"44:65:0d:56:cc:d3":2,"70:ee:50:18:34:43":3,"f4:f2:6d:93:51:f1":4,"00:16:6c:ab:6b:88":5,"30:8c:fb:2f:e4:b2":6,"00:62:6e:51:27:2e":7,"00:24:e4:11:18:a8":8,"ec:1a:59:79:f4:89":9,"50:c7:bf:00:56:39":10,"74:c6:3b:29:d7:1d":11,"ec:1a:59:83:28:11":12,"18:b4:30:25:be:e4":13,"70:ee:50:03:b8:ac":14,"00:24:e4:1b:6f:96":15,"74:6a:89:00:2e:25":16,"00:24:e4:20:28:c6":17,"d0:73:d5:01:83:08":18,"18:b7:9e:02:20:44":19,"e0:76:d0:33:bb:85":20,"70:5a:0f:e4:9b:c0":21}
#devices ={"d0:52:a8:a4:e6:46":1,"00:fc:8b:84:22:10":2,"70:ee:50:18:34:43":3,"f4:f2:6d:93:51:f1":4,"00:16:6c:ab:6b:88":5,"30:8c:fb:2f:e4:b2":6,"00:62:6e:51:27:2e":7,"00:24:e4:11:18:a8":8,"ec:1a:59:79:f4:89":9,"50:c7:bf:b1:d2:78":10,"74:c6:3b:29:d7:1d":11,"ec:1a:59:83:28:11":12,"18:b4:30:25:be:e4":13,"70:ee:50:36:98:da":14,"00:24:e4:1b:6f:96":15,"74:6a:89:00:2e:25":16,"00:24:e4:20:28:c6":17,"d0:73:d5:01:83:08":18,"18:b7:9e:02:20:44":19,"e0:76:d0:33:bb:85":20,"70:5a:0f:e4:9b:c0":21}

dm = {'0:26:29:0:77:ce': 1,
'0:3:7f:96:d8:ec': 2,
'0:c:43:3:51:be': 3,
'0:e0:4c:c9:46:31': 4, 
'0:e:f3:2c:d4:4': 5, 
'0:fc:8b:84:22:10': 6,
'34:29:8f:1c:f3:9c': 7, 
'3c:71:bf:25:c5:60': 8,
'40:31:3c:e6:77:c2': 9,
'48:46:c1:1c:46:a5': 10,
'50:32:37:b8:c7:f': 11,
'50:c7:bf:ca:3f:9d': 12,
'54:60:9:6f:32:84': 13,
'58:b3:fc:5e:ca:74': 14,
'58:ef:68:99:7d:ed': 15,
'5c:41:5a:29:ad:97': 16,
'64:16:66:2a:98:62': 17,
'68:c6:3a:ba:c2:6b': 18,
'68:c6:3a:e4:85:61': 19,
'70:ee:50:36:98:da': 20,
'74:40:be:cd:21:a4': 21,
'78:a5:dd:28:a1:b7': 22,
'7c:49:eb:22:30:9c': 23,
'7c:49:eb:88:da:82': 24,
'84:18:26:7c:1a:56': 25,
'ae:ca:6:e:ec:89': 26,
'b0:be:76:be:f2:aa': 27,
'b0:f1:ec:d4:26:ae': 28,
'b8:2c:a0:28:3e:6b': 29,
'c:2a:69:11:1:ba': 30,
'c8:3a:6b:fa:1c:0': 31,
'c:8c:24:b:be:fb': 32,
'cc:f7:35:25:af:4d': 33,
'cc:f7:35:49:f4:5': 34,
'd0:52:a8:a4:e6:46': 35,
'ec:71:db:49:af:ee': 36,
'ec:b5:fa:0:98:da': 37,
'ec:fa:bc:2e:85:5b': 38,
'f0:45:da:36:e6:23': 39,
'f4:b8:5e:68:8f:35': 40,
'fc:3:9f:93:22:62': 41
}

#for file in os.listdir(folder):
df = pd.read_csv(sys.argv[1], header=None, names=["src","Time","Length"])[["src","Time","Length"]]
df["Time"] = df["Time"].apply(lambda x: int(x))

#Replaces the MAC address of IoT devices with labels.
#for d in devices:
df["src"] = df.replace({"src": dm})

#Extracts IoT devices from the original dataset.
df = df[df['src'].astype(str).str.isdigit()]

#Groups packets into one-second windows for each IoT device.
df = df.groupby(["Time","src"]).agg(['mean', 'sum', 'std'])
df.fillna(0, inplace=True)

df.to_csv(f"{sys.argv[2]}", header=False)
sys.exit()
#Computes the statistical features for each one-second window and saves it to temporary CSV files.
g.mean().to_csv("length_avg.csv",sep=",")
g.sum().to_csv("length_sum.csv",sep=",")
g.std().to_csv("length_std.csv",sep=",")

#Creates a new data frame to store the statistical features.
df_final = pd.DataFrame()

#Populates the new data frame with statistical features and labels.
df_final["avg"] = pd.read_csv("length_avg.csv")["Length"]
df_final["n_bytes"] = pd.read_csv("length_sum.csv")["Length"]
df_final["std"] = pd.read_csv("length_std.csv")["Length"]
df_final["label"] = pd.read_csv("length_avg.csv")["src"]

#Discard NaN values.
df_final = df_final.dropna()

#Save the statistical features to a new CSV file
df_final.to_csv(str(file)+"_statistics.csv",sep=",",mode='a',index=False,header=False)

