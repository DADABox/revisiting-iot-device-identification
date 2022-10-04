from pathlib import Path
from scapy.all import load_layer, PcapReader, PcapWriter
from scapy.layers.inet import Ether, IP
from scapy.packet import raw
import sys, os
import pandas as pd


class PcapToInput(object):

    def __init__(self, filePath, numPackets=10, packetLength=250):
        self.filePath = filePath
        self.numPackets = numPackets
        self.packetLength = packetLength

    def processFile(self):
        pcap = PcapReader(self.filePath.as_posix())
        i = 0
        rawPackets = []

        try:
            for p in pcap:
                #print(p)
                if len(rawPackets) >= self.numPackets:
                    break
                p = self._removeFields(p)
                p = self._trimPacket(p)
                #print(p)
                rawPackets.append(p)
                #rawPackets[i] = raw(p)
             
                i+=1

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if len(rawPackets) < self.numPackets:
                for i in range(len(rawPackets), self.numPackets):
                    rawPackets.append(bytes([0]*self.packetLength))

        return b''.join(rawPackets)

    def _removeFields(self, packet):
        packet[Ether].src = 0
        packet[Ether].dst = 0
        packet[IP].src = 0

        return packet


    def _trimPacket(self, packet):
        packet = raw(packet)
        if len(packet) > self.packetLength:
            return packet[0:self.packetLength]

        return packet + bytes([0]*(self.packetLength - len(packet)))

    def saveRawPackets(self, saveFile, packets):
        df = pd.DataFrame({'deviceId': 1, 'packets': packets})
        df.to_csv('tmp.csv', index=False)
        #return
        with open(saveFile.as_posix(), 'wb') as f:
            for p in packets:
                f.write(p)


class PcapsToInput(object):

    def __init__(self, deviceId: int, numPackets=10, packetLength=250):
        self.deviceId = deviceId
        self.rawPackets = []
        
        self.flowCounter = 0
        self.metadata = {}

    def processDir(self, dirPath: Path):
        for path in dirPath.iterdir():
            #print(path)
            pti = PcapToInput(path)
            self.rawPackets.append(pti.processFile())
            self.flowCounter+= 1

    def saveRawPackets(self, saveFile):
        with open(saveFile.as_posix(), 'wb') as f:
            f.write(b''.join(self.rawPackets))

if __name__ == "__main__":
    #pti = PcapToInput(Path(sys.argv[1]))
    #rp = pti.processFile()
    #print(len(rp),rp[99])
    #pti.saveRawPackets(Path('tmp.raw'), rp)

    ptis = PcapsToInput(sys.argv[1])
    ptis.processDir(Path(sys.argv[2]))
    ptis.saveRawPackets(Path(f"/data/roman/flows/{sys.argv[1]}_{ptis.flowCounter}_{sys.argv[3]}.bin"))
