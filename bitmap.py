#!/usr/bin/python
# coding=utf-8

import data
import radix_sort
import numpy
import numbapro
import time
import math
from numbapro import cuda
from numba import *

tpb = 1024

@cuda.jit('void(int32[:,:],int32[:,:],int32[:,:])',target = 'gpu')
def sum(a,b,c):	
	i,j = cuda.grid(2)
	c[i][j] = a[i][j] + b[i][j]
	

def set_bit(num,off_set):
	#off_set should be range from 0 to 31.
	#The right bit refers to 0 while the left to 31
	mask = 1<<off_set
	return (num|mask)
	
def bin(s):
	#transform the integer to the type of binary code
	#return value is a string
	return str(s) if s<=1 else bin(s>>1) + str(s&1)

#step2.produce chId and literal
@cuda.jit('void(int64[:], uint32[:], int64[:])', target='gpu')
def produce_chId_lit_gpu(rid, literal, chunk_id):
	i = cuda.grid(1)
	chunk_id[i] = rid[i]/31
	literal[i] = (literal[i]|1<<31) #the left bit set to 1
	off_set = 30-rid[i]%31
	literal[i] = (literal[i]|1<<off_set)

@cuda.jit('void(int32[:], int64[:], int32, int32[:])')
def produce_flag(input_data, chunk_id, length, flag):#flag initialized to 1
	i = cuda.grid(1)
	if i<length and i>0:
		if input_data[i]==input_data[i-1] and chunk_id[i] == chunk_id[i-1]:
			flag[i] = 0

@cuda.jit('void(int32[:], int32[:], int32[:], int32, int64)')
def reduce_by_key_gpu(literal, flag, is_finish, hop, length):
	i = cuda.grid(1)
	if i < length-hop:
		if (not is_finish[i]) and (not flag[i+hop]):
			literal[i] |= literal[i+hop]
		else:
			is_finish[i] = 1


def reduce_by_key(input_data, chunk_id, literal, length):#step 3
	flag = numpy.ones(length, dtype='int32')
	stream = cuda.stream()
	d_flag = cuda.to_device(flag, stream)
	d_chunk_id = cuda.to_device(chunk_id, stream)
	d_literal = cuda.to_device(literal, stream)
	block_num = length/tpb + 1
	produce_flag[block_num,tpb](input_data, d_chunk_id, length, d_flag)
	d_flag.to_host(stream)
	print 'flag:'
	print flag
	stream.synchronize()	
	is_finish = numpy.zeros(length, dtype='int32')
	hop = 1
	while hop<32:#only 32 because the length of a word in binary form is 32
		reduce_by_key_gpu[block_num,tpb](d_literal, d_flag, is_finish, hop, length)
		hop *= 2
	
	d_literal.to_host(stream)
	d_chunk_id.to_host(stream)
	stream.synchronize()

	print '************!!!!!!!!!!!!!!!****************'
	reduced_input_data = []
	reduced_chunk_id = []
	reduced_literal =[]
	for i in xrange(length):
		if flag[i]:
			print '************!!!!!!!!!!!!!!!****************'
			reduced_input_data.appid.appendend(input_data[i])
			reduced_chunk_(chunk_id[i])
			reduced_literal.append(literal[i])

	print '************!!!!!!!!!!!!!!!****************'
	return numpy.array(reduced_input_data), numpy.array(reduced_chunk_id), reduced_literal

@cuda.jit('void(int32[:], int32[:], int64)')
def produce_head(reduced_input_data, d_head, reduced_length):
	i = cuda.grid(1)
	if i<reduced_length and i>0:
		if reduced_input_data[i]==reduced_input_data[i-1]:
			d_head[i] = 0

@cuda.jit('void(int32[:], int64[:], int64[:], int64)')
def produce_fill_gpu(d_head, d_reduced_chunk_id, reduced_chunk_id, reduced_length):
	i = cuda.grid(1)
	if i<reduced_length:
		if not d_head[i]:
			d_reduced_chunk_id[i] = reduced_chunk_id[i] - reduced_chunk_id[i-1] - 1
		
def produce_fill(reduced_input_data, reduced_chunk_id, reduced_length):#step 4
	head = numpy.ones(reduced_length, dtype='int32')
	stream = cuda.stream()
	d_head = cuda.to_device(head, stream)
	d_reduced_input_data = cuda.to_device(reduced_input_data, stream)
	
	block_num = reduced_length/tpb + 1

	produce_head[block_num,tpb](d_reduced_input_data, d_head, reduced_length)#produce head
	d_head.to_host(stream)
	stream.synchronize()
	print 'head:**************************************'
	print head
	d_reduced_chunk_id = cuda.to_device(reduced_chunk_id,stream)
	produce_fill_gpu[block_num, tpb](d_head, d_reduced_chunk_id, reduced_chunk_id, reduced_length)
	d_reduced_chunk_id.to_host(stream)
	stream.synchronize()
	#convert to int32 because the range a fill_word can describe is 0~(2^31-1)
	return numpy.array(reduced_chunk_id, dtype='int32'), head

@cuda.jit('void(int32[:], int32[:], uint32[:], int32[:], int64)')
def getIdx_gpu(fill_word, reduced_literal, index, compact_flag, length):
	i = cuda.grid(1)
	if i<length:
		index[i*2] = fill_word[i]
		index[i*2+1] = reduced_literal[i]
		if not fill_word[i]:
			compact_flag[i*2] = 0

@cuda.jit('void(uint32[:], int64[:], int64[:], uint32[:], int64)')
def scatter_index(d_index, d_compact_flag, compact_flag, out_index, reduced_length):
	i = cuda.grid(1)
	if i<2*reduced_length and compact_flag[i]:
		out_index[d_compact_flag[i]] = d_index[i]		


def getIdx(fill_word,reduced_literal, reduced_length, head, cardinality):#step 5: get index by interleaving fill_word and literal(also remove all-zeros word)
	bin_length = max(len(bin(2*reduced_length-1)),len(bin(tpb-1)))#the bit number of binary form of array length
	thread_num = numpy.int64(math.pow(2,bin_length))#Blelloch_scan need the length of scanned array to be even multiple of thread_per_block
	compact_flag = numpy.ones(thread_num, dtype='int64')
	print thread_num
	print reduced_length
	index = numpy.ones(2*reduced_length, dtype='uint32')
	d_index = cuda.to_device(index)
	d_fill_word = cuda.to_device(fill_word)
	d_reduced_literal = cuda.to_device(numpy.array(reduced_literal))
	d_compact_flag = cuda.to_device(compact_flag)
	#print fill_word

	block_num = reduced_length/tpb + 1

	getIdx_gpu[block_num, tpb](d_fill_word, d_reduced_literal, d_index, d_compact_flag, reduced_length)
	compact_flag = d_compact_flag.copy_to_host()
	print 'compact:'
	print compact_flag[0:28]

	useless_array = numpy.zeros(thread_num, dtype='int64')
	radix_sort.Blelloch_scan_caller(d_compact_flag, useless_array, 0)
	out_index_length = d_compact_flag.copy_to_host()[2*reduced_length-1] + 1
	print d_compact_flag.copy_to_host()[0:2*reduced_length]
	print out_index_length
	out_index = numpy.zeros(out_index_length, dtype='uint32')
	offset = []
	
	new_block_num = 2*reduced_length/tpb + 1

	scatter_index[new_block_num, tpb](d_index, d_compact_flag, compact_flag, out_index, reduced_length)
	#for i in out_index: 
	print head[-100:-1]
	for i in xrange(reduced_length):
		if head[i]:
			offset.append(d_compact_flag.copy_to_host()[2*i])
	#print offset
	
	key_length = numpy.zeros(cardinality, dtype='int64')
	for i in xrange(cardinality-1):
		key_length[i] = offset[i+1] - offset[i]
	key_length[cardinality-1] = out_index_length - offset[cardinality-1]
	print key_length

	return out_index, numpy.array(offset), numpy.array(key_length)	
		

'''
def get_idxlen_offset(fill_0_word,scanned_input_data):#step 6: get offsets and key
	scanned_length = len(fill_0_word)
	tmp_array = [1]*scanned_length
	for i in xrange(scanned_length):
		if fill_0_word[i] != 0:
			tmp_array[i] += 1

	#6.1: index length for each key
	key_idx_length = [tmp_array[0]]
	for i in xrange(1,scanned_length):
		if scanned_input_data[i] == scanned_input_data[i-1]:
			key_idx_length[-1] += tmp_array[i]
		else:
			key_idx_length.append(tmp_array[i])
	#6.2: offset for each key in the whole index
	offset = [0 for i in key_idx_length] 
	for i in xrange(1,len(key_idx_length)):
		offset[i] = offset[i-1] + key_idx_length[i-1]
	return key_idx_length,offset
'''

if __name__ == '__main__':
	path = 'data.txt'	#file path
	attr_dict,attr_values,attr_value_NO = data.openfile(path)
	total_row = len(attr_values[0])
	input_data = numpy.array(attr_values[0])
	length = input_data.shape[0]
	rid = numpy.arange(0,length)
	print input_data
	#step1 sort
	radix_sort.radix_sort(input_data,rid)
	cardinality = input_data[-1]+1 #
	print 'rid:\n',rid
	literal = numpy.zeros(length, dtype = 'uint32')
	chunk_id = numpy.zeros(length, dtype = 'int64')
	stream = cuda.stream()
	d_rid = cuda.to_device(rid, stream)
	d_chunk_id = cuda.to_device(chunk_id, stream)
	d_literal = cuda.to_device(literal, stream)
	#step2 produce chunk_id and literal
	produce_chId_lit_gpu[length/tpb+1, tpb](d_rid, d_literal, d_chunk_id)
	d_rid.to_host(stream)
	d_chunk_id.to_host(stream)
	d_literal.to_host(stream)
	stream.synchronize()

	#step3 reduce by key(value, chunk_id)
	reduced_input_data,	reduced_chunk_id, reduced_literal = reduce_by_key(input_data, d_chunk_id, d_literal, length)
	reduced_length = reduced_input_data.shape[0]

	#step4 produce 0-Fill word
	fill_word, head = produce_fill(reduced_input_data, reduced_chunk_id, reduced_length)

	for i in xrange(reduced_length):
		print reduced_input_data[i], reduced_chunk_id[i], bin(reduced_literal[i]), fill_word[i]
	
	#step 5 & 6: get index by interleaving 0-Fill word and literal(also remove all-zeros word)
	out_index, offset, key_length = getIdx(fill_word,reduced_literal, reduced_length, head, cardinality)
	
