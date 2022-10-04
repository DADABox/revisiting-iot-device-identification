This set of scripts trains and evaluates models for a statistical data created from one second window of packets. 

Description of files:

- `extractFromPcap.sh` takes a directory with pcap files as an input and extract mac address, time, and packet length using tshark
- `extract\_features\_chronological.py` takes a file produced by previous script as an input and for each one second window computes statistical information (mean, sum, and standard deviation)
- `testClass.py` trains and evaluates models on data produced by previous script
