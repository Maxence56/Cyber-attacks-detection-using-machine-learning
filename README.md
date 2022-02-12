# Cyber-attacks-detection-using-machine-learning

Projet de recherche au département informatique de IMT Atlantique ou j’ai évalué la performance de plusieurs algorithmes de classification supervisés appliqués au dataset réduit CICIDS2017 généré en simulant un trafic bénin et diverses attaques subites par des postes clients dans un réseau informatique.

## Contenu des scripts

### Machine_learning

Permet de réaliser le preprocessing nécessaire sur les fichiers du dataset, de les séparer en fichiers d'entraînement et de test et de les concaténer s'il le faut pour avoir un fichier unique

Permet d'entraîner et de tester différents algorithmes de classification sur le dataset: 

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Decision Forest
- Naive Bayes
- K-nearest Neighbors

Permet de réaliser la méthode d'Elbow sur le dataset afin de déterminer le nombre de classes à l'aide de l'algorithme k-means

Des fonctions en bas du fichier facilitent le lancement des différentes fonctions proposées. 

### Neuralnetwork_TSNE

Permet de réaliser le preprocessing nécessaire sur les fichiers du dataset, de les séparer en fichiers d'entraînement et de test et de les concaténer s'il le faut pour avoir un fichier unique

Permet de définir les propriétés d'un réseau de neurones, de l'entraîner et de le tester. 

Permet d'avoir une représentation bidimensionnelle de la sortie de l'avant-dernière couche du réseau de neurones lors de la classification à travers un algorithme de réduction de dimensions (TSNE) qui permet de visualiser la séparation des différentes classes d'attaque lors de la classification. 

Des fonctions en bas du fichier facilitent le lancement des différentes fonctions proposées. 
