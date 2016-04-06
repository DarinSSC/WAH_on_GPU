#!/usr/bin/python
# coding=utf-8

#Implementation of radix sort algorithmn on GPU
#STEP 1

import numpy
import numbapro
import time
import math
from numbapro import cuda
from numba import *

TPB_MAX = 1024
ATTR_CARD_MAX = int(math.pow(2,32))-1


def bin(s):
	#transform the integer to the type of binary code
	#return value is a string
	return str(s) if s<=1 else bin(s>>1) + str(s&1)

min

@cuda.jit('void(int64[:], int64[:], int64, int64)',target='gpu')
def reduce_phase(zero_list, one_list, hop, thread_num):
    i = cuda.grid(1)
    if i%(2*hop) == (2*hop-1):
        zero_list[i] += zero_list[i-hop]
        one_list[i] += one_list[i-hop]
    if hop == thread_num/2:
        if i == thread_num-1:
            one_list[i] = 0
            zero_list[i] = 0


@cuda.jit('void(int64[:], int64[:], int64, int64)',target='gpu')
def downsweep_phase(zero_list, one_list, hop, base):
    i = cuda.grid(1)
    if i%(2*hop) == (2*hop-1):
        zero_list[i-hop], zero_list[i] = zero_list[i], zero_list[i-hop]+zero_list[i]
        one_list[i-hop], one_list[i] = one_list[i], one_list[i-hop]+one_list[i]
    cuda.syncthreads()
    if hop==1:
        one_list[i] += base

def Blelloch_scan_caller(d_zero_list, d_one_list, base):#zero_list uses 1 to represent 0, 0 to represent 1; one_list on the contrary
    thread_num = d_zero_list.shape[0]
    block_num = thread_num/TPB_MAX
    print 'scan block_num: %d'%block_num
    #step1. reduce
    hop = 1
    while hop < thread_num:
        reduce_phase[block_num,TPB_MAX](d_zero_list, d_one_list, hop, thread_num)
        hop *= 2
	
    #step2. downsweep
    hop/= 2
    while hop > 0:
        downsweep_phase[block_num,TPB_MAX](d_zero_list, d_one_list, hop, base)
        hop /= 2


@cuda.jit('void(int64[:], int64[:])',target='gpu')
def sum_reduction(zero_list, tmp_out):
    bw = cuda.blockDim.x
    bx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    shared_list = cuda.shared.array(shape = (TPB_MAX), dtype = int64)
   
    i = bx*bw + tid
    shared_list[tid] = zero_list[i]
    cuda.syncthreads()

    hop = bw/2
    while hop > 0:
        if tid < hop:
            shared_list[tid] = shared_list[tid] + shared_list[tid+hop]
        cuda.syncthreads()
        hop /= 2
    if tid == 0:
        tmp_out[bx] = shared_list[0]

@cuda.jit('void(int32[:], int64, int32, int64[:], int64[:])', target='gpu')
def get_list(arr, length, iter_order, zero_list, one_list):
    i = cuda.grid(1)
    if i < length:
        one_list[i] = (arr[i]>>iter_order)%2
        zero_list[i] = 1-one_list[i]
#        if flag:
#            one_list[i] = 1
#            zero_list[i] = 0
#        else:
#            zero_list[i] = 1
#            one_list[i] = 0
    else:
        one_list[i] = 0
        zero_list[i] = 0

@cuda.jit('void(int32[:], int32[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64)', target='gpu')
def array_adjust(arr, d_arr,rid, d_rid, zero_list, one_list, d_zero_list, d_one_list, length):
    i = cuda.grid(1)
    if i<length:
        if zero_list[i] == 1:
            arr[d_zero_list[i]] = d_arr[i]
            rid[d_zero_list[i]] = d_rid[i]
        else:
            arr[d_one_list[i]] = d_arr[i]
            rid[d_one_list[i]] = d_rid[i]

def radix_sort(arr, rid):
    length = numpy.int64(len(arr))
    bin_length = max(len(bin(length-1)),len(bin(TPB_MAX-1)))#the bit number of binary form of array length
    thread_num = numpy.int64(math.pow(2,bin_length))
    block_num = max(thread_num/TPB_MAX,1)

    print 'length: %d'%length
    print 'bin_length: %d'%bin_length
    print 'thread_num: %d'%thread_num
    print 'block_num: %d'%block_num

    stream = cuda.stream()
    one_list = numpy.zeros(shape=(thread_num), dtype='int64')
    zero_list = numpy.zeros(shape=(thread_num), dtype='int64')

    iter_num = len(bin(ATTR_CARD_MAX))
    print 'iter_num: %d'%iter_num
    for i in range(iter_num):
        print '***************************'
        print 'iteration_%d:'%i
        print arr
        d_arr = cuda.to_device(arr, stream)
        d_rid = cuda.to_device(rid, stream)
        d_zero_list = cuda.to_device(zero_list,stream)
        d_one_list = cuda.to_device(one_list,stream)
        get_list[block_num, TPB_MAX](arr, length, i, d_zero_list, d_one_list)#get one_list and zero_list
        d_one_list.to_host(stream)
        d_zero_list.to_host(stream)
        stream.synchronize()
        print 'zero_list:'
        print zero_list
        print 'one_list'
        print one_list
        
        base_reduction_block_num = block_num
        base_reduction_block_size = TPB_MAX
        
        print 'base_reduction_block_num: %d'%base_reduction_block_num
        tmp_out = numpy.zeros(base_reduction_block_num, dtype='int64')
        d_tmp_out = cuda.to_device(tmp_out, stream)
        sum_reduction[base_reduction_block_num, base_reduction_block_size](d_zero_list, d_tmp_out)
        d_tmp_out.to_host(stream)
        stream.synchronize()
        base = 0 #base for the scan of one_list
        for j in xrange(base_reduction_block_num):
            base += tmp_out[j]
        print 'base: %d'%base

        #then do scanning(one_list and zero_list at the same time)
        print 'begin scan'
        Blelloch_scan_caller(d_zero_list, d_one_list, base)
        
        print 'scan finished'
        print
        #adjust array elements' position
        print 'begin adjust'
        print 'zero_list:'
        print zero_list
        array_adjust[block_num,TPB_MAX](arr, d_arr, rid, d_rid, zero_list, one_list, d_zero_list, d_one_list, length)
        print arr
        print

if __name__ == '__main__':
	n = 1025
	#arr = numpy.array([n-1-i for i in xrange(n)])
	arr = numpy.random.randint(1000,size=(n))
	print arr
	print '**********************'
	rid = numpy.arange(0,n)
	radix_sort(arr, rid)
	print arr
	print rid
