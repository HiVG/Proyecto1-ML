# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:25:15 2020

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.neural_network import MLPRegressor, MLPClassifier

def lasso(x, y):
    sv = LassoCV(normalize=True)
    sv.fit(x,y)
    print("Mejor alpha usando LassoCV: %f" % sv.alpha_)
    print("Mejor valor usando LassoCV: %f" % sv.score(x, y))
    coef = pd.Series(sv.coef_, index=x.columns)
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Modelo Lasso para selección de variables")
    plt.show()
    
        
def rn(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=1)
    mlp = MLPClassifier(hidden_layer_sizes=(25,), activation='relu', solver='adam', alpha=0.01, random_state=1)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    print("------Redes Neuronales-------")
    print(mlp.score(x_test, y_test))
    
def rl(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)
    regres=LogisticRegression(C=1.0, solver= 'lbfgs', penalty= 'l2', random_state=1)
    regres.fit(X_train,y_train)
    pred=regres.predict(X_test)
    print("----------regresión logística---------")
    print(pred)
    cm=confusion_matrix(y_test,pred)
    print("----------la matriz de confusión---------")
    print(cm)
    score=regres.score(X_test,y_test)
    print("----------resultado---------")
    print(score)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
    scores=cross_val_score(regres,X,y,cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
    print("----------métricas---------")
    print(classification_report(y_test,pred))
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize=(10,10))
    sns.heatmap(df_cm,annot=True, cmap='coolwarm')
    
def pre(df):
    drinks = df["drinks"].values
    cat=[]
    for i in drinks:
        if (i>=5):
            cat.append(1)
        else:
            cat.append(0)
    print(cat)
    cat = pd.DataFrame(data=cat,columns=["cat"])
    d = pd.concat([df,cat],axis=1)
    d.drop(["drinks"],axis=1,inplace=True)
    #print(d)
    return d       

   
    
def reemplazar(df):
    df.replace('?',-99999,inplace=True)
    return df

def main():
    df = pd.read_csv("bupa.data", names =['mcv','alkphos','sgpt','sgot','gamma','drinks','selector'])
    df = pre(df)
    y = df["cat"]
    listDrop = ["cat"]
    df = df.drop(listDrop, 1)
    print(y)
    print(df)
    
    df1 = pd.read_csv("lung-cancer.data", header=None)
    df1 = reemplazar(df1)
    print(df1)
    y1 = df1.iloc[:,0]
    print(y1)
    x = df1.iloc[:,1:]
    print(x)
    
    listita=[]
    listita.append(df1.iloc[:,19])
    listita.append(df1.iloc[:,53])
    listita.append(df1.iloc[:,20])
    listita.append(df1.iloc[:,35])
    listita.append(df1.iloc[:,54])
    listita.append(df1.iloc[:,11])
    data = pd.DataFrame(data=listita)
    data= data.transpose()
    print(data)
    
    
    
    #rl(df, y)
   
    #lasso(x,y1)
    
    #rl(x,y1)
    
    rl(data,y1)
    
    #rn(x,y1)
    
    rn(data,y1)
   

if __name__=="__main__":
    main()