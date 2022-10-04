#!/bin/bash

pcap=$1
pcapFile=$(basename $pcap)
mac=$2
devId=$3

destDir="flows/$pcapFile"

# this script requires that you have mono installed (framework for executing .NET executables on Linux)
# and SpltiCap.exe which splits input pcap into separate TCP/UDP flows

mono SplitCap.exe -r $pcap -p 1014 -o $destDir

