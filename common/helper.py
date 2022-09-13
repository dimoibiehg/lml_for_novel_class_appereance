import math 
import sys, os
import _pickle as cPickle
import pickle

def highestPowerof2(n):
 
    p = int(math.log(n, 2))
    return int(pow(2, p))

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def store_in_pickle(file_address, data):
    try:
        p = cPickle.Pickler(open(file_address,"wb")) 
        p.fast = True 
        p.dump(data)
        return True
    except Exception as error:
        print(error)
        return False
    
def retrieve_from_pickle(file_address):
    try:
        p = cPickle.Unpickler(open(file_address,"rb")) 
        seqs_list = p.load()
        return seqs_list
    except:
        return None