‘‘‘Importing the function’’’
from sklearn import tree
from sklearn.neural_network import MLPClassifier
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB as p
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
import math
import numpy as np
import pandas as pd

#‘‘‘K-Nearest Neighbors Function’’’


def KNN(x_training,y_training,x_testing,y_testing):
  print("\nKNN")
  clf=KNeighborsClassifier(n_neighbors=9)
  clf.fit(x_training,y_training)
  cvs=cross_val_score(clf,x_testing,y_testing,cv=5)
  result=clf.predict(x_testing)
  cmknn=confusion_matrix(y_testing,result)
  print("Confusion_Matrix\n\n",cmknn)
  print("Accuracy =",((cmknn[0][0]+cmknn[1][1])/len(y_testing))*100)
  print("Mean=",cvs.mean())

#‘‘‘NaiveB Function’’’

def NaiveB(x_training,y_training,x_testing,y_testing):
  print("\n\nNaiveBaysian")
  gnb=p()
  gnb.fit(x_training,y_training)
  y_pre=gnb.predict(x_testing)
  cm=confusion_matrix(y_testing,y_pre)
  print("Confusion_Matrix\n",cm)
  print("Accuracy=",((cm[0][0]+cm[1][1])/len(y_testing)*100))

#‘‘‘Decision Tree Function’’’

def DecisionTree(x_training,y_training,x_testing,y_testing):
  print("\n\nDecision Tree")
  clf=tree.DecisionTreeClassifier()
  clf=clf.fit(x_training,y_training)
  y_predict=clf.predict(x_testing)
  CM=confusion_matrix(y_testing,y_predict)
  accuracy=sum(CM.diagonal())/len(y_test)
  print("Confusion_Matrix\n",CM)
  print("Accuracy=",accuracy*100)

#‘‘‘Artificial Neural Network Function’’’

def NeuralNetwork(x_training,y_training,x_testing,y_testing):
  print("\n\n Artificial Neural Network")
  
  clf = MLPClassifier(solver='lbfgs', alpha=1e-
  5,hidden_layer_sizes=(5,3),random_state=1) #default hidden layer=100
  
  clf.fit(x_training,y_training) #The default solver ‘adam’ works pretty well on
  relatively large datasets
  y_predict=clf.predict(x_testing)
  cm=confusion_matrix(y_testing,y_predict)
  print("Confusion matrix\n\n",cm)
  print("Accuracy=",sum(cm.diagonal())/len(x_testing)*100)

#‘‘‘Support Vector Machine Function’’’

def SVM(x_training,y_training,x_testing,y_testing):
  print("\n\nSVM")
  clf = svm.SVC(gamma='auto')
  clf.fit(x_training,y_training)
  y_predict=clf.predict(x_testing)
  cm=confusion_matrix(y_testing,y_predict)
  print("Confusion matrix\n\n",cm)
  print("Accuracy=",sum(cm.diagonal())/len(x_testing)*100)

#‘‘‘Algorithm’’’

def algorithm(x_train,y_train,x_test,y_test):
  acc=[]
  NaiveB(x_train,y_train,x_test,y_test,acc)
  KNN(x_train,y_train,x_test,y_test,acc)
  DecisionTree(x_train,y_train,x_test,y_test,acc)
  NeuralNetwork(x_train,y_train,x_test,y_test,acc)
  SVM(x_train,y_train,x_test,y_test,acc)

#‘‘‘Importing csv file’’’

d=pd.read_csv("breast_cancer_weka_dataset.csv")
c=d.copy()
x_train=c.loc[:350,:"mitosis"]
y_train=c.loc[:350,"class"]
x_test=c.loc[350:,:"mitosis"]
y_test=c.loc[350:,"class"]

#‘‘‘Applying function’’’

NaiveB(x_train,y_train,x_test,y_test)
KNN(x_train,y_train,x_test,y_test)
DecisionTree(x_train,y_train,x_test,y_test)
NeuralNetwork(x_train,y_train,x_test,y_test)
SVM(x_train,y_train,x_test,y_test)
