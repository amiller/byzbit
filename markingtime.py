"""
Andrew Miller <amiller@cs.ucf.edu> <amiller@dappervision.com>

An illustration of how Bitcoin measures time according to cumulative work.
Experiment described in this post:
https://bitcointalk.org/index.php?topic=98986.msg1109747#msg1109747

Dependencies:
     BitcoinTools: https://github.com/gavinandresen/bitcointools
"""

from bsddb3.db import * # imports DB*
from block import scan_blocks
from util import create_env
import numpy as np
from bitarray import bitarray
from binascii import hexlify
import pylab

db_dir = "/home/amiller/.bitcoin/testdb"
db_env = create_env(db_dir)

def hash_value(h):
    # A 'smoother' version of the number of leading zero bits
    return 256 - np.log2(float(h))

def hash_value2(h):
    # Count the number of leading zeros
    if h == 0: return 256
    if h > (1<<256)-1: return 0
    c = 0
    while not (h & (1<<255)):
        h <<= 1
        c += 1
    return c

def nbits_to_target(bits):
    # Convert the Bitcoin nBits field (difficulty measure) to target
    coeff = bits & 0x00ffffff
    expon = ((bits & 0xff000000) >> 24) - 3
    target = coeff * 2 ** (8 * expon)
    return target

def target_to_work(target):
    # Expected number of hashes needed to reach the target
    return (2**256) / float(target)

def scan():
    # Return the height, work, and hash value for each block
    result = []
    def scan_callback(block_data):
        height = block_data['nHeight']
        work = target_to_work(nbits_to_target(block_data['nBits']))
        value = hash_value(long(hexlify(block_data['hash256'][::-1]), 16))
        result.append((height,work,value))
        return True

    scan_blocks(db_dir, db_env, scan_callback)
    return np.array(result, dtype='f8')[::-1]

def main():
    dv = scan()

    # Plot the hash values and work per block, ordered by time (block height)
    pylab.figure(1)
    pylab.clf()
    pylab.scatter(dv[:,0], dv[:,2], s=0.1, label='Hash Values (bits)')
    pylab.plot(dv[:,0], np.log2(dv[:,1]),'r',linewidth=2, label='Difficulty (minimum value)')
    pylab.title('Hash Value vs Time')
    pylab.ylabel('Hash Value (zero bits) (log2(hash))')
    pylab.xlabel('Time (blocks)')
    pylab.legend(loc=4)

    work = np.cumsum(dv[:,1])
    pylab.figure(2)
    pylab.clf()
    pylab.scatter(work, dv[:,2], s=0.1, label='Hash Values (bits)')
    pylab.plot(work, np.log2(dv[:,1]),'r',linewidth=2, label='Difficulty (minimum value)')
    pylab.title('Hash Value vs Work')
    pylab.ylabel('Hash Value (zero bits) (log2(hash))')
    pylab.xlabel('Cumulative Work (est. hashes computed)')
    pylab.legend(loc=4)
