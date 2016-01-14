#!/usr/bin/python
# coding=utf-8

import data
import numpy
import numbapro
import time
from numbapro import cuda

@cuda.jit('void(int32[:],int32,int32,int32)',target = 'gpu')
def bitonic_builder(d_list, length, hop, inner_hop):
	tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	order_tag = (tid / (2 * hop)) % 2
	if (order_tag == 0):#ascending order
		if ((tid / inner_hop) % 2 == 0):
			if (d_list[tid] > d_list[tid + inner_hop]):
				d_list[tid],d_list[tid + inner_hop] = d_list[tid + inner_hop], d_list[tid]
	else:#descecnding order
		if ((tid / inner_hop) % 2 == 0):
			if (d_list[tid] < d_list[tid + inner_hop]):
				d_list[tid],d_list[tid + inner_hop] = d_list[tid + inner_hop], d_list[tid]

def builder_caller(d_list, length):
	hop = 1
	while hop < length / 2:
		print "hop: %d"%hop
		inner_hop = hop

		while inner_hop > 0:
			print("inner_hop=%d:  "%inner_hop)
			block_size = 2 * inner_hop
			if block_size > 1024:#每个block最多分配1024个线程,否则出错(why???)
				block_size = 1024

			block_num = length / block_size
			bitonic_builder[block_num, block_size](d_list, length, hop, inner_hop)
			#cudaDeviceSynchronize()
			for i in d_list:
				print i,
			print ''
			#cudaDeviceSynchronize()
			inner_hop /= 2
		hop *= 2

@cuda.jit('void(int32[:],int32,int32)',target = 'gpu')
def bitonic_sort(d_bitonic_list, length, hop):
	if (hop <= cuda.blockDim.x):
		idx = cuda.blockDim.x * cuda.blockIdx.x * 2 + cuda.threadIdx.x;
	else:
		idx = cuda.blockDim.x * (cuda.blockIdx.x % (hop / cuda.blockDim.x)) + cuda.threadIdx.x + (cuda.blockIdx.x / (hop / cuda.blockDim.x)) * hop * 2;
	#idx = cuda.blockDim.x * cuda.blockIdx.x * 2 + cuda.threadIdx.x
	if idx+hop < length:
		if d_bitonic_list[idx] > d_bitonic_list[idx + hop]: 
			d_bitonic_list[idx], d_bitonic_list[idx + hop] = d_bitonic_list[idx + hop], d_bitonic_list[idx]

def bitonic_sort_caller(d_bitonic_list, length): 
	hop = 1
	while hop * 2 < length:
		hop *= 2
	block_size = hop
	if block_size > 1024:
		block_size = 1024
	

	while hop > 0:
		block_size = hop
		if block_size>1024:
			block_size = 1024
		block_num = length/(2*block_size)

		print "hop=%d: "%hop
		print "block_num: %d, "%block_num,
		print "block_size: %d"%block_size
		bitonic_sort[block_num, block_size](d_bitonic_list, length, hop)
		#cudaDeviceSynchronize()
		#cudaPrint<<<1, 1>>>(d_bitonic_list, length)
		#cudaDeviceSynchronize()
		for i in d_bitonic_list:
			print i,
		print ''
		hop /= 2

def errorDetect(order_list):
	iswrong = 0
	length = len(order_list)
	for i in range(length)[1:-1]:
		if order_list[i] < order_list[i-1]:
			print 'wrong at '+ str(i)
			iswrong = 1
			break;
	if not iswrong:
		print 'Right answer!'

if __name__ == '__main__':
	#numbapro.check_cuda()
	time0 = time.time()
	length =256*1024
	d = numpy.random.randint(100000,size = [length])
	print d
	builder_caller(d,length)
	print '****************sort start**************'
	bitonic_sort_caller(d,length)
	time1 = time.time()
	delta_time = time1-time0
	print 'delta_time:%d'%delta_time
	errorDetect(d)
