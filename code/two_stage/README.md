This is a set of scripts using a two stage classifier. 

When extracting features from pcap files a joy utility from Cisco needs to be installed [https://github.com/cisco/joy](https://github.com/cisco/joy)

Description of files:

- `deviceMap.py` maps MAC addresses to device IDs
- `getDictionary.py` extracts data from a Multinomial Naive Bayes Classifier
- `getFeatures.py` extracts features from pcap files. Separate pcap file for each 1 hour window needs to be created. 
- `testiMNB.py` classfies data using Naive Bayes Multinomial (NBM)
- `testRF.py` classifies data using Random Forest (RF)
