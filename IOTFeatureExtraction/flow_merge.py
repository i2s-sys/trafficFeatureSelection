#python 3.5.2
#pcap文件的合并
from scapy.all import *
from scapy.layers.inet import TCP, IP, UDP

# 遍历整个文件夹 接收一个文件夹路径作为参数。
def each_file(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        if os.path.isfile(child):
            each_pkt(child)
        elif os.path.isdir(child) and allDir != '.ipynb_checkpoints':
            each_file(child)
        else:
            continue
    print('pcap_merge,down!!!') # 表示数据包合并完成。


#将packet包进行合并 处理单个pcap文件。
def each_pkt(filename):
    global packets
    #loading the pcap file
    dpkt=rdpcap(filename)    #- 加载指定文件的pcap数据包。
    for packet in dpkt:
        packets.append(packet)

        
# flow合并
def flow_merge(filepath):
    each_file(filepath)
    # wrpcap函数将合并后的流量数据写入一个新的pcap文件，该文件命名为"pcap_merge.pcap"
    wrpcap(str(filepath)+'/pcap_merge.pcap',packets)


# 主函数，仅有1个参数，就是文件路径 调用 流合并函数 来处理文件路径中的数据包
if __name__ == '__main__':
    packets=[]
    filepath='test_data/yourthing'
    flow_merge(filepath)
