This set of scripts trains and evaluates a neural network with dense layers and a random forest classifier on data extracted from TCP/UDP flows.

The flows were extracted from PCAP files using a [joy](https://github.com/cisco/joy) utility from Cisco.

Description of the files:

- `testNN.py` creates and evaluates models using neural network
- `testRF.py` creates and evaluates models using Random Forest Classifier
