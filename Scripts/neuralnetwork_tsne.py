#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:59:36 2021

@author: m20elfat
"""
from sklearn.metrics import confusion_matrix , accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras import Model
import prettyplot as pp
import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers import Dense
import time
import pandas as pd
global filelist
filelist = ['Bot','BruteForceXssSql','DDos','Dosmulti','FtpSsh','Portscan','normal','Infiltration']
global file
file = filelist[0]
global directory
#directory = "D:\\Telechargements\\dataset\\"
directory = "/home/elfatihi/Téléchargements/"

global labels
labels = {'Bot':['BENIGN','Bot'],'BruteForceXssSql':['BENIGN','Brute Force','XSS','Sqlinjection'],'DDos':['BENIGN','DDoS'],'Dosmulti':['BENIGN','DoS Hulk','DoS GoldenEye','DoS Slowhttptest','DoS slowloris','Heartbleed'],'FtpSsh':['BENIGN','FTP-Patator','SSH-Patator'],'Portscan':['BENIGN','PortScan'],'Infiltration':['BENIGN','Infiltration'],'normal':['BENIGN']}


""" Meme fonction que sur l'autre fichier """

def fonction_preprocess(All=True , normalize=False , lefile="Bot" , labels_uniques= False, fichier_unique = True):
    global directory
    global filelist
    global labels
    start_time = time.time()
    if All:
        liste_des_fichiers = [k for k in filelist]
    else : liste_des_fichiers = [lefile]
    
    for fichier in liste_des_fichiers:
        
        df = pd.read_csv(directory+fichier+'.csv')
        
        if labels_uniques :
            
            numeros = { ' Label': {'BENIGN':0 ,'Brute Force':1,'XSS':2,'Sqlinjection':3,'DDoS':4,'DoS Hulk':5,'DoS GoldenEye':6,'DoS Slowhttptest':7,'DoS slowloris':8,'Heartbleed':9,'FTP-Patator':10,'SSH-Patator':11,'PortScan':12,'Infiltration':13,'Bot':14}}
        
        else:
        
            numeros={' Label':{}}
            
            for k in range(len(labels[fichier])):
                numeros[' Label'][labels[fichier][k]] = k
            #numeros = { ' Label': {'BENIGN':0 ,'Brute Force':1,'XSS':2,'Sqlinjection':3}}
        
        df=df.replace(numeros)
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
        c=dfn.shape[1]
        
        if normalize:
            for k in range (0,c-1):
                if (np.max(dfn[::,k]) == np.min(dfn[::,k])):
                    dfn[::,k] = 1.0
                else:
                    dfn[::,k] = ((dfn[::,k]) - np.min(dfn[::,k])) / (np.max(dfn[::,k]) - np.min(dfn[::,k]))
            
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
        
        if fichier_unique :
            if labels_uniques:
                df.to_csv(directory+'processedall'+fichier+'.csv',index=False)
                print(fichier+' Terminé!')
                print( "Temps d'execution: %s" %(time.time() - start_time) )
            else:
                df.to_csv(directory+'processed'+fichier+'.csv',index=False)
                print(fichier+' Terminé!')
                print( "Temps d'execution: %s" %(time.time() - start_time) )
                
            
        else:
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


    
def concat(sauf=['normal'],type_fichiers=['train','test'][0]):
    global directory
    global filelist
    start_time = time.time()
    
    liste_des_fichiers = [ k for k in filelist if k not in sauf]
    start_time = time.time()
    data = pd.concat([pd.read_csv(directory+type_fichiers+'all'+f+'.csv') for f in liste_des_fichiers])
    data.to_csv(directory+type_fichiers+'allattacks'+'.csv',index=False)
    print("Done !")
    print( "Temps d'execution: %s" %(time.time() - start_time) )

  
#fonction qui sauvegarde le réseau de neurones entrainé

def save_model(model,fichier):
    global directory
    # saving model
    json_model = model.to_json()
    open(directory+'model_architecture_'+fichier+'train.json', 'w').write(json_model)
    # saving weights
    model.save_weights(directory+'model_weights_'+fichier+'train.h5', overwrite=True)

#fonction qui charge le réseau de neurones entrainé

def load_model(fichier):
    # loading model
    model = model_from_json(open(directory+'model_architecture_'+fichier+'train.json').read())
    model.load_weights(directory+'model_weights_'+fichier+'train.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
    
#fonction qui definit les propriétés du réseau de neurones
    
def baseline_model(types_of_flows):
    # Create model here
    model = Sequential()
    model.add(Dense(512, input_dim = 78, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(256, input_dim = 78, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    
    #a renomer lors d'un prochain entrainement en 'avant derniere'
    
    model.add(Dense(64, activation = 'relu',name="layer_3"))
    model.add(Dense(types_of_flows, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model    

#fonction qui permet d'entrainer le réseau de neurones

def buildnn(All=False , lefile="allattacks",sauf=['normal']):
    global filelist
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefile]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'train'+fichier+'.csv')
    
        y = df.iloc[:,78]
        
        print(fichier)
        print(df.groupby(y).size())
    
        
        X = df.iloc[:,0:78]
        
       
        y = pd.get_dummies(y)
        
        types_of_flows = len(y.columns)
    
    
        model = baseline_model(types_of_flows)


        model.fit(X, y, epochs=50, batch_size=50, verbose=1)
        
        
        # save
        save_model(model,fichier)
    
        print( "Temps d'entrainement: %s" %(time.time() - start_time) )
        
        if False:
            print(" Model Parameters ")
            for k in range(3):
                print("Layer "+str(k))
                print(model.layers[k].get_weights())
    
    #affiche les poids des neurones
    
    
    
#classification a l'aide d'un réseau de neurones

def usenn(All=True , lefile="Bot",saveplots = True,sauf=['normal']):
    global filelist
    global labels
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefile]
    
    for fichier in liste_des_fichiers:
        start_time = time.time()
        df = pd.read_csv(directory+'test'+fichier+'.csv')
    
        y = df.iloc[:,78]
        
        print(fichier)
        print(df.groupby(y).size())
    
        
        X = df.iloc[:,0:78]
        
    

        model = load_model(fichier)
    
        # predictions
        y_pred = model.predict_classes(X, verbose=1)
        
    
        
        
        print(accuracy_score(y_pred, y))
        
        print(confusion_matrix(y, y_pred))

        
        pp.plot_confusion_matrix_from_data(y,y_pred,labels[fichier], algo='Neural Network',fichier=fichier,directory=directory,saveplots=saveplots)
        
        print( "Temps d'execution: %s" %(time.time() - start_time) )


def tsne_nn(All=True , lefichier="FtpSsh",sauf=["normal"], saveplots = True):
    global filelist
    global directory
    start_time = time.time()
    
    
    if All :
        liste_des_fichiers = [ k for k in filelist if k not in sauf]
    else :
        liste_des_fichiers = [lefichier]
    
    for fichier in liste_des_fichiers:
        print(fichier)
        df = pd.read_csv(directory+'test'+fichier+'.csv')
        y = df.iloc[:,78]
        print(df.groupby(y).size())
        X = df.iloc[:,0:78]
        
       
        y = pd.get_dummies(y)
    
        model = load_model(fichier)
        
        #on utilise le nom donné  à la couche
        #a renomer lors d'un prochain entrainement
        
        layer_name = "layer_3"
        intermediate_layer_model = Model(inputs=model.input,
                                                outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(X.to_numpy()).numpy()
        
        
        tsne = TSNE(n_components=2,learning_rate=200,verbose=1)
        tsne_features = tsne.fit_transform(intermediate_output)
        print( "Temps d'execution: %s" %(time.time() - start_time) )
        
        #on rajoute au dataframe deux nouvelles colonnes contenant les coordonnées TSNE
        
        df['x'] = tsne_features[:,0]
        df['y'] = tsne_features[:,1]
        
        #on remplace les labels entiers par les noms des attaques
        
        numeros={' Label':{}}
        for k in range(len(labels[fichier])):
                numeros[' Label'][labels[fichier][k]] = k
        
        numeros[' Label'] = {v: k for k, v in numeros[' Label'].items()}
        
        
        df = df.replace(numeros)
        
        print(df[' Label'])
        
        sns.lmplot(x="x",y = "y", data=df,
                fit_reg=False,
                legend=True,
                size=9,
                hue=' Label',
                scatter_kws={"s":40, "alpha":0.3})
        
        
        
        if saveplots:
            
            plt.savefig(directory+"TSNE"+fichier+".png")
           
        plt.show()
        
        print( "Temps d'execution: %s" %(time.time() - start_time) )


#meme fonction que creation_fichiers_pour_algos_classification():
def creation_fichiers_pour_reseau_de_neurones():
    fonction_preprocess(All=True , normalize=True , lefichier="Bot" , labels_uniques= False, sans_train_test = False , sauf=['normal'])
    #Permet de traiter les différents fichiers et de génerer des fichiers train et test

#meme fonction que creation_fichier_general_pour_algos_classification():    
def creation_fichier_general_pour_reseau_de_neurones():
    fonction_preprocess(All=True , normalize=True , lefichier="Bot" , labels_uniques= True, sans_train_test = False , sauf=['normal'])
    concat(sauf=['normal'],type_fichiers=['train','test','processed'][0])
    concat(sauf=['normal'],type_fichiers=['train','test','processed'][1])
    #permet de créer 2 fichiers: train et test correspondant à toutes les attaques du dataset


def entrainement_reseau_de_neurones_fichiers_simples():
    buildnn(All=True , lefile="allattacks",sauf=['normal'])
    
    
def entrainement_reseau_de_neurones_fichier_general():
    buildnn(All=False , lefile="allattacks",sauf=['normal'])
    
    
def utilisation_reseau_de_neurones_fichiers_simples():
    usenn(All=True , lefile="Bot",saveplots = True, sauf=['normal'])
    
    
def utilisation_reseau_de_neurones_fichier_general():
    usenn(All=False, lefile="allattacks",saveplots = True, sauf=['normal'])
    

def tsne_reseau_de_neurones_fichiers_simples(saveplots=True):
    tsne_nn(All=True , lefichier="FtpSsh",sauf=["normal"], saveplots = saveplots)
    
    
def tsne_reseau_de_neurones_fichier_general(saveplots=True):
    tsne_nn(All=False, lefichier="allattacks",sauf=["normal"], saveplots = saveplots)
    