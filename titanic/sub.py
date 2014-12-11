import csv as csv
import numpy as np 

csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()

data = []
num_surv = 0
num_pass = 0

for row in csv_file_object:
	data.append(row)

data = np.array(data)
#You must do this to enable any of the fancy operations.

num_surv = np.sum(data[0::,1].astype(np.float))
num_pass = np.size(data[0::,1].astype(np.float))

print num_surv/num_pass

num_male = data[0::,4]=="male"
num_male = data[num_male,1].astype(np.float)
num_female = data[0::,4] == "female"
num_female = data[num_female,1].astype(np.float)

print "Female survivors" 
print np.sum(num_female)
print "Male survivors"
print np.sum(num_male)
print num_surv


test_ptr = open('test.csv', 'rb')
csv_test = csv.reader(test_ptr)
header = csv_test.next()

temp = []

predict_ptr = open('genderbasedmodel.csv', 'wb')
csv_predict = csv.writer(predict_ptr)
csv_predict.writerow(["PassengerId", "Survived"])

for row in csv_test:
	
	if(row[3]=="female"):
		csv_predict.writerow([row[0], '1'])
	else:
		csv_predict.writerow([row[0], '0'])
		
'''
We always close the file pointer, not the data. Be careful.
'''

test_ptr.close()
predict_ptr.close()		
			
