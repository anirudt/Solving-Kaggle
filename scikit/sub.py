import numpy as np
from sklearn import grid_search
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold

workDir = r'C:\Transit\Kaggle\\'

# Read data
train = np.genfromtxt(open(workDir + 'train.csv','rb'), delimiter=',')
target = np.genfromtxt(open(workDir + 'trainLabels.csv','rb'), delimiter=',')
test = np.genfromtxt(open(workDir + 'test.csv','rb'), delimiter=',')

# This takes in the input without compromising the first row

pca = PCA(n_components=12,whiten=True)
train = pca.fit_transform(train)
test = pca.transform(test)

'''
Basically, a PCA is done to reduce higher dimensional problem to a
lower dimensional problem.
'''
print "hello"

'''
preparing the parameters
'''
C_range = 10.0 ** np.arange(6.5,7.5,0.20)
gamma_range = 10.0 ** np.arange(-1.5,0.5,0.20)

print "hello"
params = dict(gamma=gamma_range,C=C_range)
cvk = StratifiedKFold(target,3) #no. of folds of cross validation
print "hello"
classifier = SVC()
print "hello"
clf = grid_search.GridSearchCV(classifier,param_grid=params,cv=cvk) # SVM Classifier buildup.
print "hello"
clf.fit(train,target)
#print("The best classifier is: ",clf.best_estimator_) 

scores = cv.cross_val_score(clf.best_estimator_, train, target, cv=30)
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

#answer = 0.92784 for n_components = 12 [initial value]
#answer = 0.95202 for n_components = 12 and C_range, gamma_range as 6.5, 7.5, 0.25, and -1.5, 0.5, 0.2

result = clf.best_estimator_.predict(test)
i = np.arange(1,9001);
i = np.transpose(i)
result = np.transpose(result)
result = np.concatenate([i,result])
print result
result = result.reshape(2,9000)
result = np.transpose(result)
np.savetxt(workDir + 'result.csv', result, delimiter=',', fmt='%d')