from typing import overload
from os import path
import json
from common.goal import *

class Stream:

    def __init__(self, stream_len, target_types = [TargetType.LATENCY, TargetType.PACKETLOSS, TargetType.ENERGY_CONSUMPTION],
                                    target_names = ["latency", "packetloss", "energyconsumption"], stream_addr = "./data/total", stream_file_base_name = "dataset_with_all_features",
                data_order = None):
        self.__base_addr = stream_addr
        self.cycles_num = stream_len
        self.__file_base_name = stream_file_base_name
        self.__target_names = target_names
        self.__target_types = target_types
        self.current_cyle = 1
        self.data_order = data_order
    def read_current_cycle(self):
        if(self.current_cyle > self.cycles_num):
            exit("end of stream")
        if(self.data_order is None):
            current_stream = self.__load_raw_features_qualities(self.current_cyle)
        else:
            current_stream = self.__load_raw_features_qualities(self.data_order[self.current_cyle - 1])
        self.current_cyle += 1
        return current_stream
    
    def read(self, cycle_num: int):
        specified_stream = self.__load_raw_features_qualities(cycle_num)
        return specified_stream


    def __flatten(self, l):
        return [e for sublist in l for e in sublist]


    def __load_raw_data(self, cycle_num:int):
        r = []
        with open(path.join(self.__base_addr, f'{self.__file_base_name}{cycle_num}.json'), 'r') as f:
            r.append(json.load(f))

        return r


    def __load_raw_data_split(self, cycle_num:int, target_name:str):
        data = self.__load_raw_data(cycle_num)

        # Features, Latency
        return 	[c['features'] for c in data], [c[target_name] for c in data], [c['verification_times'] for c in data]

        # Features, Packet loss, Latency, Energy consumption, Verification times
                # [c['packetloss'] for c in data], \
                
                # [c['energyconsumption'] for c in data], \
                # [c['verification_times'] for c in data]



    def __load_raw_features_qualities(self, cycle_num:int):
        
        target_vals = {}
        for i in range(len(self.__target_types)):
            features, target, verfs_time = self.__load_raw_data_split(cycle_num, self.__target_names[i])
            target_vals[self.__target_types[i]] = self.__flatten(target)
        return self.__flatten(features), target_vals, verfs_time
    

def fetch_stream(stream_len, target_types = [TargetType.LATENCY, TargetType.PACKETLOSS, TargetType.ENERGY_CONSUMPTION],
                                            target_names = ["latency", "packetloss", "energyconsumption"], data_order = None,
                                            stream_addr = "./data/FirstTry"):
    # if(with_drift):
    return Stream(stream_len, target_types= target_types,\
                target_names= target_names, \
                # stream_addr = "./data/sudden_locals_20_distributed")
                stream_addr = stream_addr, data_order = data_order)