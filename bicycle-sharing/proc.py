import sklearn
import pandas as pd
import csv
from datetime import datetime
from sklearn import svm
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

#File IO and Feature Selection
df = pd.DataFrame.from_csv("train.csv", sep = ",")
num_elements = len(df['count'])

df_dt = pd.read_csv("train.csv")


features = df.drop(['count', 'casual', 'registered'],1)
# Machine Learning
print features.dtypes

y = df['count']

clf = svm.SVR()
#Optimal value of k0
feat_new = SelectKBest(chi2, k=2).fit_transform(features,y)
clf.fit(feat_new, y)

# Testing phase

df_test = pd.read_csv("test.csv", sep=",")
print df_test

feat_test = df_test.drop('datetime', 1)

#TODO: Select the 2 best features of the test set as well
#print np.linalg.norm(y_pred-y[10000:10886])*1.0/886
y_pred = pd.Series(np.round(clf.predict(feat_test)))
np.savetxt('output1.csv',y_pred,delimiter=',')

y_write = list(df_test['datetime'])
#print y_write

import csv
f = open("output_f.csv", "wb")
writer = csv.writer(f)
y_final = [['datetime', 'count']]
i = 0
while(i<len(y_write)):
	y_final.append([y_write[i], y_pred[i]])
	i+=1
'''
writer.writerows(y_final)
clf.fit(features[0:6000], y[0:6000])
y_1 = np.round(clf.predict(features[6000:10000]))
print np.linalg.norm(y_1-y[6000:10000])*1.0/4000

k0 = 2
while(k0<=8):
	print k0
	feat_new = SelectKBest(chi2, k=k0).fit_transform(features,y)
	clf.fit(feat_new[0:6000], y[0:6000])
	y_p = clf.predict(feat_new[6000:10000])
	print np.linalg.norm(y_p-y[6000:10000])*1.0/4000
	k0+=1
4.13564967689
2
4.064813469
3
4.08150353865
4
4.1495628385
5
4.14485683729
6
4.14365359024
7
4.14057051296
8
4.13552351691
'''	