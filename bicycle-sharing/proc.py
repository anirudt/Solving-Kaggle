import sklearn
import pandas as pd
import csv
from datetime import datetime
from sklearn import svm
import numpy as np

#File IO and Feature Selection
df = pd.DataFrame.from_csv("train.csv", sep = ",")
num_elements = len(df['count'])

df_dt = pd.read_csv("train.csv")


features = df.drop(['count', 'casual', 'registered'],1)
# Machine Learning
print features.dtypes

y = df['count']

clf = svm.SVR()
clf.fit(features, y)

# Testing phase

df_test = pd.read_csv("test.csv", sep=",")
print df_test

feat_test = df_test.drop('datetime', 1)
#print np.linalg.norm(y_pred-y[10000:10886])*1.0/886
y_pred = pd.Series(np.round(clf.predict(feat_test)))
np.savetxt('output1.csv',y_pred,delimiter=',')

y_write = list(df_test['datetime'])
print y_write

import csv
f = open("output_f.csv", "wb")
writer = csv.writer(f)
y_final = [['datetime', 'count']]
i = 0
while(i<len(y_write)):
	y_final.append([y_write[i], y_pred[i]])
	i+=1
writer.writerows(y_final)
#np.savetxt('output.csv',y_write,delimiter=',')
#y_final = pd.concat([y_write, y_pred], axis=1)
#print y_final
#np.savetxt('output.csv',y_final,delimiter=',')