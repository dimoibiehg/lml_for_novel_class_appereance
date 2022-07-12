import math 
import sys, os

def highestPowerof2(n):
 
    p = int(math.log(n, 2))
    return int(pow(2, p))

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__