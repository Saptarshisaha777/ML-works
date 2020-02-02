from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score
data=load_iris()

label_name=data['target_names']
feature_name=data['feature_names']
print(label_name,feature_name)
X=data.data
Y=data.target
print(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.25,random_state=0)
gnb=GaussianNB()
model=gnb.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
#accuracy=gnb.score(Y_test,Y_pred)
#accuracy=gnb.score(X,Y)
print(accuracy_score(Y_test,Y_pred))


svm1=svm.SVC(kernel='linear')
model1=svm1.fit(X_train,Y_train)
Y_pred1=model1.predict(X_test)
accuracy1=svm1.score(X,Y)
print(accuracy1)

svm2=svm.SVC(kernel='poly',degree=8)
model2=svm2.fit(X_train,Y_train)
Y_pred2=model2.predict(X_test)
accuracy2=svm2.score(X,Y)
print(accuracy2)