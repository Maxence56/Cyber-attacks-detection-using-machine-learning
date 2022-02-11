#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:26:24 2021

@author: Maxence ELFATIHI
"""

"""machine learning"""
import time
import os.path
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix , accuracy_score

import prettyplot as pp
global random_state
random_state = 17
global filelist
filelist = ['Bot','BruteForceXssSql','DDos','Dosmulti','FtpSsh','Portscan','normal','Infiltration']
global file
file = filelist[0]
global directory
#directory = "D:\\Telechargements\\dataset\\"
directory = "/home/elfatihi/Téléchargements/"
""" En fonction du système dexploitation, bien definir le directory qui va contenir les datasets et dans lequel vont etre sauvegardés les fichiers générés """

global labels
labels = {'Bot':['BENIGN','Bot'],'BruteForceXssSql':['BENIGN','Brute Force','XSS','Sqlinjection'],'DDos':['BENIGN','DDoS'],'Dosmulti':['BENIGN','DoS Hulk','DoS GoldenEye','DoS Slowhttptest','DoS slowloris','Heartbleed'],'FtpSsh':['BENIGN','FTP-Patator','SSH-Patator'],'Portscan':['BENIGN','PortScan'],'Infiltration':['BENIGN','Infiltration'],'normal':['BENIGN']}
global kernel
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel = kernels[1]


""" 
    librarie prettyplot: 
        
        
        


    fonction de preprocessing :
    
    All: Tous les fichiers dans filelist sauf les fichiers dans "sauf" 
    normalize: True:permet de normaliser (les resultats de la methode elbow varient en fct de la normalisation)
    lefichier: pour un traitement d'un fichier unique si All vaut False
    labels_uniques: True:permet de remplacer les labels par une numerotation unique à utiliser pour la concatenation avec All= True.
    sans_train_test: True: permet de ne pas split en fichier train et test. utile pour le kmeans

    La division des fichiers est réalisée en amont pour gagner du temps


"""

def fonction_preprocess(All=True , normalize=True , lefichier="Bot" , labels_uniques= False, sans_train_test = False , sauf=['normal']):
    global directory
    global filelist
    global labels
    start_time = time.time()
    if All:
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else : liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        print(fichier)
        
        df = pd.read_csv(directory+fichier+'.csv')
        
        if labels_uniques :
            
            numeros = { ' Label': {'BENIGN':0 ,'Brute Force':1,'XSS':2,'Sqlinjection':3,'DDoS':4,'DoS Hulk':5,'DoS GoldenEye':6,'DoS Slowhttptest':7,'DoS slowloris':8,'Heartbleed':9,'FTP-Patator':10,'SSH-Patator':11,'PortScan':12,'Infiltration':13,'Bot':14}}
        
        else:
        
            numeros={' Label':{}}
            
            for k in range(len(labels[fichier])):
                numeros[' Label'][labels[fichier][k]] = k
            #exempl
            #numeros = { ' Label': {'BENIGN':0 ,'Brute Force':1,'XSS':2,'Sqlinjection':3}}
            #on remplace les labels avec des numeros 
        df=df.replace(numeros)
        
        #nettoyage du dataset
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        dfn = df.to_numpy()
        dfn = [x for x in dfn if 'NaN' not in x]
        dfn = [x for x in dfn if 'NAN' not in x]
        dfn = [x for x in dfn if 'INF' not in x]
        dfn = [x for x in dfn if 'NINF' not in x]
        dfn = [x for x in dfn if 'Infinity' not in x]
        dfn = [x for x in dfn if '-Infinity' not in x]
        dfn = np.array([np.array(xi) for xi in dfn])
        dfn = dfn.astype(float)
        
        # print(np.isfinite(dfn).all())
        # print(np.isfinite(dfn).all())
        #permet de verifier si les données sont finies
        c=dfn.shape[1]
        
        #normalisation
        if normalize:
            for k in range (0,c-1):
                if (np.max(dfn[::,k]) == np.min(dfn[::,k])):
                    dfn[::,k] = 1.0
                else:
                    dfn[::,k] = ((dfn[::,k]) - np.min(dfn[::,k])) / (np.max(dfn[::,k]) - np.min(dfn[::,k]))
            
            #precaution supplementaire
            l=[]
            for i in range(dfn.shape[0]) :
                for j in range(dfn.shape[1]-1):
                    if( dfn[i,j] > 1 or  dfn[i,j] <0 ):
                        l.append(i)
                        break
            l = tuple(l)
            dfn = np.delete(dfn, l, axis=0)
        
        
        
        dfr = pd.DataFrame(dfn)
        dfr.columns = df.columns.values
        
        df = dfr
        
        #sauvegarde des fichiers obtenus
        
        if sans_train_test :
            if labels_uniques:
                df.to_csv(directory+'processedall'+fichier+'.csv',index=False)
                print(fichier+' Terminé!')
                print( "Temps d'execution: %s" %(time.time() - start_time) )
            else:
                df.to_csv(directory+'processed'+fichier+'.csv',index=False)
                print(fichier+' Terminé!')
                print( "Temps d'execution: %s" %(time.time() - start_time) )
                
            
        else:
            #split des fichiers
            
            df['split'] = np.random.randn(df.shape[0], 1)
            
            msk = np.random.rand(len(df)) <= 0.7
            
            train = df[msk]
            test = df[~msk]
        
            if labels_uniques:
                train.iloc[:,:-1].to_csv(directory+'trainall'+fichier+'.csv',index=False,float_format='%.15f')
                test.iloc[:,:-1].to_csv(directory+'testall'+fichier+'.csv',index=False,float_format='%.15f')
            
            else:
                train.iloc[:,:-1].to_csv(directory+'train'+fichier+'.csv',index=False,float_format='%.15f')
                test.iloc[:,:-1].to_csv(directory+'test'+fichier+'.csv',index=False,float_format='%.15f')
            
            print(fichier+' Terminé!')
            print( "Temps d'execution: %s" %(time.time() - start_time) )

def concat(sauf=['normal'],type_fichiers=['train','test','processed'][0]):
    global directory
    global filelist
    start_time = time.time()
    
    liste_des_fichiers = [ k for k in filelist if k not in sauf]
    start_time = time.time()
    data = pd.concat([pd.read_csv(directory+type_fichiers+'all'+f+'.csv') for f in liste_des_fichiers])
    data.to_csv(directory+type_fichiers+'allattacks'+'.csv',index=False)
    print("Done !")
    print( "Temps d'execution: %s" %(time.time() - start_time) )


    
def allalgos():
    logistic()
    svm()
    decisiontree()
    randomforest()
    naivebayes()
    
    if False :
        knn()




 #saveplots:permet de sauvegarder les figures
def logistic(All=False , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    global filelist
    global labels
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        #fichier d'entrainement
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y_train = df.iloc[:,78]
        
        print(fichier)
        print('Training')
        
        #statistiques sur les fichiers utilisés 
        print(df.groupby(y_train).size())
    
        
        X_train = df.iloc[:,0:78]
        
       
        
        clf = LogisticRegression(solver='lbfgs', max_iter=100,).fit(X_train, y_train)
        
        
        #fichier de test
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y_test = df.iloc[:,78]
        print('Prediction')
        print(df.groupby(y_test).size())
        X_test = df.iloc[:,0:78]

        #predictions
        y_pred = clf.predict(X_test)
        
        #performances de l'algo
        
        print('Logistic Regression')
        print(fichier)
        print(accuracy_score(y_pred, y_test))
        print(confusion_matrix(y_test,y_pred))
        pp.plot_confusion_matrix_from_data(y_test=y_test, predictions=y_pred, list_labels=labels[fichier], algo='Logistic Regression',fichier=fichier,directory=directory,saveplots=saveplots)
        print( "Temps d'execution: %s" %(time.time() - start_time) )
    
def svm(All=False , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    global filelist
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y_train = df.iloc[:,78]
        
        print(fichier)
        print('Training')
        print(df.groupby(y_train).size())
    
        
        X_train = df.iloc[:,0:78]
        
       
        
       
        svclassifier = SVC(kernel=kernel,degree=30,max_iter=100).fit(X_train, y_train)
        
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y_test = df.iloc[:,78]
        print('Prediction')
        print(df.groupby(y_test).size())
        X_test = df.iloc[:,0:78]
        
       
        y_pred = svclassifier.predict(X_test)
        
        
        print('SVM')
        print(fichier)
        print(accuracy_score(y_pred, y_test))
        print(confusion_matrix(y_test,y_pred))
        pp.plot_confusion_matrix_from_data(y_test,y_pred,labels[fichier], algo='SVM',fichier=fichier,directory=directory,saveplots=saveplots)
        print( "Temps d'execution: %s" %(time.time() - start_time) )

def decisiontree(All=False , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    global filelist
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y_train = df.iloc[:,78]
        
        print(fichier)
        print('Training')
        print(df.groupby(y_train).size())
    
        
        X_train = df.iloc[:,0:78]
        

        dtc = DecisionTreeClassifier().fit(X_train,y_train)
        
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y_test = df.iloc[:,78]
        print('Prediction')
        print(df.groupby(y_test).size())
        X_test = df.iloc[:,0:78]
        

        y_pred = dtc.predict(X_test)


        print('Decision Tree')
        print(fichier)
        print(accuracy_score(y_pred, y_test))
        print(confusion_matrix(y_test,y_pred))
        pp.plot_confusion_matrix_from_data(y_test,y_pred,labels[fichier], algo='Decision Tree',fichier=fichier,directory=directory,saveplots=saveplots)
        print( "Temps d'execution: %s" %(time.time() - start_time) )
    
    
   
def randomforest(All=False , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    global filelist
    global labels
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y_train = df.iloc[:,78]
        
        print(fichier)
        print('Training')
        
        
        
        
        print(df.groupby(y_train).size())
        
        
        
        X_train = df.iloc[:,0:78]
        
       
       
        rfc = RandomForestClassifier(n_estimators=100, random_state=random_state).fit(X_train,y_train)
        
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y_test = df.iloc[:,78]
        print('Prediction')
        print(df.groupby(y_test).size())
        X_test = df.iloc[:,0:78]
        
       
        y_pred = rfc.predict(X_test)
        
        
        print('Random forest')
        print(fichier)
        print(accuracy_score(y_pred, y_test))
        print(confusion_matrix(y_test,y_pred))
        pp.plot_confusion_matrix_from_data(y_test,y_pred,labels[fichier], algo='Random forest',fichier=fichier,directory=directory,saveplots=saveplots)
        print( "Temps d'execution: %s" %(time.time() - start_time) )
        
        
def naivebayes(All=False , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    global filelist
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y_train = df.iloc[:,78]
        
        print(fichier)
        print('Training')
        print(df.groupby(y_train).size())
    
        
        X_train = df.iloc[:,0:78]
        
       
       
       
        gnb = GaussianNB().fit(X_train,y_train)
        
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y_test = df.iloc[:,78]
        print('Prediction')
        print(df.groupby(y_test).size())
        X_test = df.iloc[:,0:78]
        
       
        
        y_pred = gnb.predict(X_test)
        
        print('NaiveBayes')
        print(fichier)
        print(accuracy_score(y_pred, y_test))
        print(confusion_matrix(y_test,y_pred))
        pp.plot_confusion_matrix_from_data(y_test,y_pred,labels[fichier], algo='NaiveBayes',fichier=fichier,directory=directory,saveplots=saveplots)
        print( "Temps d'execution: %s" %(time.time() - start_time) )



def knn(All=False , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    
    #Attention! Execution tres longue 
    
    
    global filelist
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y_train = df.iloc[:,78]
        
        print(fichier)
        print('Training')
        print(df.groupby(y_train).size())
    
        
        X_train = df.iloc[:,0:78]
        
       

        knnc = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
        
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y_test = df.iloc[:,78]
        print('Prediction')
        print(df.groupby(y_test).size())
        X_test = df.iloc[:,0:78]
        
   
        y_pred = knnc.predict(X_test)

        print('K-Nearest Neighbors')
        print(fichier)
        print(accuracy_score(y_pred, y_test))
        print(confusion_matrix(y_test,y_pred))
        #y_test, predictions,list_labels,algo,fichier,directory,
        pp.plot_confusion_matrix_from_data(y_test,y_pred,labels[fichier], algo='K-Nearest Neighbors',fichier=fichier,directory=directory,saveplots=saveplots)
        print( "Temps d'execution: %s" %(time.time() - start_time) )
    

def kmeans(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=True):
    global filelist
    global directory
    print('KMeans')
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        #il faut utiliser les fichiers non splités
        df = pd.read_csv(directory+'processed'+fichier+'.csv')

        print(fichier)
        y = df.iloc[:,78]
        print(df.groupby(y).size())
    
        
        X = df.iloc[:,0:78]
        
        elbow=[]
        
        abscisse = range(1,7)
        
        for k in abscisse:
            kmeanModel = KMeans(k).fit(X)
            elbow.append(kmeanModel.inertia_)
            
        plt.figure(figsize=(16,8))
        plt.plot(abscisse, elbow,'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method  '+fichier)
        if saveplots:
            plt.savefig(directory+"KMeans"+fichier+".png")
        plt.show()
        print( "Temps d'execution: %s" %(time.time() - start_time) )
    
    
def creation_fichiers_pour_algos_classification():
    fonction_preprocess(All=True , normalize=True , lefichier="Bot" , labels_uniques= False, sans_train_test = False , sauf=['normal'])
    #Permet de traiter les différents fichiers et de génerer des fichiers train et test

    
def creation_fichier_general_pour_algos_classification():
    fonction_preprocess(All=True , normalize=True , lefichier="Bot" , labels_uniques= True, sans_train_test = False , sauf=['normal'])
    concat(sauf=['normal'],type_fichiers=['train','test','processed'][0])
    concat(sauf=['normal'],type_fichiers=['train','test','processed'][1])
    #permet de créer 2 fichiers: train et test correspondant à toutes les attaques du dataset


def creation_fichiers_pour_kmeans(normalisation = False):
    fonction_preprocess(All=True , normalize=normalisation , lefichier="Bot" , labels_uniques= False, sans_train_test = True , sauf=['normal'])
    #permet de traiter les différents fichiers sans les split pour les utiliser dans l'algo kmeans


def lancement_algos_classification_fichiers_simples(saveplots=True):
    logistic(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)
    svm(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)
    decisiontree(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)
    randomforest(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)
    naivebayes(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)
    
    if False:#attention long !
        knn(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)


def lancement_algo_classification_fichier_general(saveplots=True):
    logistic(All=False , lefichier="allattacks",sauf=['normal'],saveplots=saveplots)
    svm(All=False , lefichier="allattacks",sauf=['normal'],saveplots=saveplots)
    decisiontree(All=False , lefichier="allattacks",sauf=['normal'],saveplots=saveplots)
    randomforest(All=False, lefichier="allattacks",sauf=['normal'],saveplots=saveplots)
    naivebayes(All=False , lefichier="allattacks",sauf=['normal'],saveplots=saveplots)
    
    if False:   #attention long !
        knn(All=False , lefichier="all",sauf=['normal'],saveplots=saveplots)

      
def lancement_kmeans_fichiers_simples(saveplots=True):
    kmeans(All=True , lefichier="FtpSsh",sauf=['normal'],saveplots=saveplots)
    
      
def creation_fichier_general_pour_kmeans(normalisation = False):
    fonction_preprocess(All=True , normalize=normalisation , lefichier="Bot" , labels_uniques= True, sans_train_test = True , sauf=['normal'])
    concat(sauf=['normal'],type_fichiers=['train','test','processed'][2])

    
def lancement_kmeans_fichier_general(saveplots=True):
    kmeans(All=False , lefichier="allattacks",sauf=['normal'],saveplots=saveplots)
   
    #attention long !
    
    