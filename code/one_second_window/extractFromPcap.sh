dir=$1

for f in $(find $dir -name "*.pcap"); do 
  tshark -r $f -T fields -e eth.src -e frame.time_epoch -e frame.len -E separator=, >> data.csv; 
done
