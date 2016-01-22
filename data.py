import string

def openfile(path):
	file_ob = open(path)
	try:
		#attribute names are in the 1st line
		attr_name = file_ob.readline().strip('\n')
		attr_list = string.split(attr_name,',')
		attr_num = len(attr_list)
		#creat a dictionary to map the attr_name to attributes
		attr_dict = {}
		attr_values = [[] for i in xrange(attr_num)]
		for attr in attr_list:
			attr_dict[attr] = set([])#usi00ng set to avoid repetition
		for line in file_ob.readlines():
			line = line.strip('\n')
			tmp_list = string.split(line,',')
			for i in xrange(len(tmp_list)):
				attr_dict[attr_list[i]].add(tmp_list[i])
				attr_values[i].append(tmp_list[i])
	finally:	
		file_ob.close()
	
	rec_num = len(attr_values[0])#the number of records
	#get cardinaities for the attributes
	attr_card = [len(attr_dict[attr]) for attr in attr_list]
	#number attr-values for each attribute
	attr_value_NO = [{} for i in xrange(attr_num)]
	for i in xrange(attr_num):
		j = 0
		for attr_value in attr_dict[attr_list[i]]:
			attr_value_NO[i][attr_value] = j
			j += 1
	#transform source data(string value) to numbered data(integer value)
	for i in xrange(attr_num):
		for j in xrange(rec_num):
			attr_values[i][j] = attr_value_NO[i][attr_values[i][j]]
	return attr_dict,attr_values,attr_value_NO


if __name__ == '__main__':
	print '********attr_list*************:'
	print attr_list
	print '********attr_dict*************:'
	for item in attr_dict.items():
		print item
	print '********attr_card*************:'
	print attr_card
	print '********attr_value_NO*********:'
	for adict in attr_value_NO:
		print '---------------------------'
		for item in adict.items():
			print item
	
	#write transformed values to a.txt
	f = open('a.txt','w')
	for i in xrange(rec_num):
		for j in xrange(attr_num):
			f.write(str(attr_values[j][i])+',')
		f.write('\n')
	f.close()
	
	
