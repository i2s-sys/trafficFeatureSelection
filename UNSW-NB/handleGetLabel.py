import pandas as pd


def extract_second_column_elements(file_path):
    data = pd.read_csv(file_path)

    second_column_elements = data.iloc[:, -1].unique()

    print(list(second_column_elements))

file_path = 'UNSW_NB15_training-set.csv'
extract_second_column_elements(file_path)

# proto_type = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp', 'ipv6-frag', 'cftp', 'wsn', 'pvp', 'wb-expak', 'mtp', 'pri-enc', 'sat-mon', 'cphb', 'sun-nd', 'iso-ip', 'xtp', 'il', 'unas', 'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe', 'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp', 'eigrp', 'ax.25', 'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp', 'zero', 'rvd', 'mobile', 'narp', 'fc', 'pipe', 'ipcomp', 'ipv6-no', 'sat-expak', 'ipv6-opts', 'snp', 'ipcv', 'br-sat-mon', 'ttp', 'tcf', 'nsfnet-igp', 'sprite-rpc', 'aes-sp3-d', 'sccopmce', 'sctp', 'qnx', 'scps', 'etherip', 'aris', 'pim', 'compaq-peer', 'vrrp', 'iatp', 'stp', 'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp', 'crtp', 'sps', 'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip', 'micp', 'encap', 'ifmp', 'tp++', 'a/n', 'ipv6', 'i-nlsp', 'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx', 'ippc', 'visa', 'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt', 'ggp', 'ip', 'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp', 'emcon', 'igp', 'nvp', 'pup', 'xnet', 'chaos', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'xns-idp', 'leaf-1', 'leaf-2', 'rdp', 'irtp', 'iso-tp4', 'netblt', 'trunk-2', 'cbt']
# service = ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc']
# state = ['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no']
# lable = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']






