import numpy as np 
import csv as csv

csv_train = csv.reader(open('train.csv', 'rb'))
header = csv_train.next()

data=[]
for row in csv_train:
	data.append(row)
data = np.array(data)
'''
This anaylsis is supposed to be based on Gender[4], Class[2], Fare[9], to predict Survival[1]

1. Binning of the fare ranges

'''
fare_threshold = 40
data[data[0::,9].astype(np.float)>=fare_threshold,9] = fare_threshold - 1

num_class = len(np.unique(data[0::,2]))
fare_size = 10
num_fare = fare_threshold/fare_size

survival_table = np.zeros((2, num_class, num_fare))

for i in xrange(num_class):
	for j in xrange(num_fare):
		num_female = data[(data[0::,4]=="female") & (data[0::,2].astype(np.float)==i+1) & (data[0::,9].astype(np.float)>=j*fare_size) & (data[0::,9].astype(np.float) <(j+1)*fare_size),1]
		num_male = data[(data[0::,4]=="male") & (data[0::,2].astype(np.float)==i+1) & (data[0::,9].astype(np.float)>=j*fare_size) & (data[0::,9].astype(np.float) <(j+1)*fare_size),1]
		survival_table[0,i,j] = np.mean(num_female.astype(np.float))
		survival_table[1,i,j] = np.mean(num_male.astype(np.float))


survival_table[survival_table!=survival_table] = 0

print survival_table		

survival_table[survival_table>=0.5] = 1
survival_table[survival_table<0.5] = 0

test_ptr = open('test.csv', 'rb')
csv_test = csv.reader(test_ptr)
header = csv_test.next()

predict_ptr = open('genderclassmodel.csv', 'wb')
csv_predict = csv.writer(predict_ptr)
csv_predict.writerow(["PassengerId", "Survived"])
'''
Test file has Class[1], Gender[3], and Fare[8]

'''

for row in csv_test:
	for j in xrange(num_fare):
		try:
			row[8] = float(row[8])

		except:
			bin_fare = 3 - float(row[1])
			break
		if(row[8]>fare_threshold):
			bin_fare = num_fare - 1
			break
		if((row[8]>=j*fare_size) & (row[8]<(1+j)*fare_size)):
			bin_fare = j		
			break
	if(row[3]=="female"):
		csv_predict.writerow([row[0], int(survival_table[0,float(row[1])-1,bin_fare])])
	else:
		csv_predict.writerow([row[0], int(survival_table[1,float(row[1])-1,bin_fare])])
test_ptr.close()
predict_ptr.close()
