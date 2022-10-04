This set of scripts trains model using raw packets from TCP/UDP flows.

They work as follows:

- `processPcap.sh` takes pcap file as an input and uses SplitCap to extract independent TCP/UDP flows
- `PcapToInput.py` takes a directory containing separate TCP/UDP flows as an input and procudes binary files containing formatted data (250x10) that can be used as input for a 2D Convolutional layer.
- `ModelBuilder.py` creates and evaluates models using data produced by PcapToInput.py
