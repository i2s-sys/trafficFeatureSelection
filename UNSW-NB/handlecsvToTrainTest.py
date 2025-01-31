import csv

from sklearn.model_selection import train_test_split


def get_col_types():
    protocol_type = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp', 'ipv6-frag', 'cftp', 'wsn', 'pvp', 'wb-expak', 'mtp', 'pri-enc', 'sat-mon', 'cphb', 'sun-nd', 'iso-ip', 'xtp', 'il', 'unas', 'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe', 'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp', 'eigrp', 'ax.25', 'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp', 'zero', 'rvd', 'mobile', 'narp', 'fc', 'pipe', 'ipcomp', 'ipv6-no', 'sat-expak', 'ipv6-opts', 'snp', 'ipcv', 'br-sat-mon', 'ttp', 'tcf', 'nsfnet-igp', 'sprite-rpc', 'aes-sp3-d', 'sccopmce', 'sctp', 'qnx', 'scps', 'etherip', 'aris', 'pim', 'compaq-peer', 'vrrp', 'iatp', 'stp', 'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp', 'crtp', 'sps', 'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip', 'micp', 'encap', 'ifmp', 'tp++', 'a/n', 'ipv6', 'i-nlsp', 'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx', 'ippc', 'visa', 'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt', 'ggp', 'ip', 'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp', 'emcon', 'igp', 'nvp', 'pup', 'xnet', 'chaos', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'xns-idp', 'leaf-1', 'leaf-2', 'rdp', 'irtp', 'iso-tp4', 'netblt', 'trunk-2', 'cbt']
    service_type = ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc']
    state_type = ['CLO','ACC','FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no']
    # label_type = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
    return protocol_type,service_type,state_type
def handle_data():
    protocol_type,service_type,state_type = get_col_types()
    # with open('conn_log_labeled.csv', 'r') as data_source:
    #     csv_reader = csv.reader(data_source)
    #     header = next(csv_reader)
    #     data = [row for row in csv_reader]
    # train_data, test_data = train_test_split(data, test_size=0.3)
    # with open('origin_train_data.csv', 'w', newline='') as train_file:
    #     csv_writer = csv.writer(train_file)
    #     csv_writer.writerow(header)
    #     csv_writer.writerows(train_data)
    # # 写入测试数据
    # with open('origin_test_data.csv', 'w', newline='') as test_file:
    #     csv_writer = csv.writer(test_file)
    #     csv_writer.writerow(header)
    #     csv_writer.writerows(test_data)
    # 从测试集开始
    source_file = 'UNSW_NB15_training-set2.csv'
    handled_file = 'train_data2.csv'  # write to csv file
    data_file = open(handled_file, 'w', newline='')
    csv_writer = csv.writer(data_file)
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        next(csv_reader)
        for row in csv_reader:
            row[1] = protocol_type.index(row[1])
            row[2] = service_type.index(row[2])
            row[3] = state_type.index(row[3])
            csv_writer.writerow(row)
        data_file.close()
    test_source_file = 'UNSW_NB15_testing-set2.csv'
    test_handled_file = 'test_data2.csv'  # write to csv file
    test_data_file = open(test_handled_file, 'w', newline='')
    test_csv_writer = csv.writer(test_data_file)
    with open(test_source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        next(csv_reader)
        for row in csv_reader:
            row[1] = protocol_type.index(row[1])
            row[2] = service_type.index(row[2])
            row[3] = state_type.index(row[3])
            test_csv_writer.writerow(row)
        test_data_file.close()
    print('pre process completed!')
handle_data()