import numpy as np
import sys
import load_datasets
#import NaiveBayes # importer la classe du classifieur bayesien
#import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés
import classifieur as classifier
from datetime import datetime
"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

if __name__ == '__main__':
    # Initialisez vos paramètres

    train_ratio = 0.7

    # Initialisez/instanciez vos classifieurs avec leurs paramètres

    # Charger/lire les datasets
    if train_ratio <= 0 or train_ratio >= 1:
        print('Veuillez saisir une valeur de ratio supérieure à 0 et inférieure à 1')
    else:
        train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)

        list_train_ratio = [0.7, 0.9]
        list_dataset = ['iris', 'wine', 'abalone']
        list_tech = ['knn', 'nb']
        list_algo = ['implementation', 'scikit']

        for train_ratio_ele in list_train_ratio:
            print('\nTraining ratio {0}\n'.format(train_ratio_ele))
            for dataset in list_dataset:
                time_before_excecution = datetime.now()
                print("\nCurrent Time =", time_before_excecution)

                if dataset == 'iris':
                    print('\n--------------1)--dataset {0} ---------------'.format(dataset))
                    train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
                if dataset == 'wine':
                    print('\n--------------2)--dataset {0} ---------------'.format(dataset))
                    train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
                if dataset == 'abalone':
                    print('\n--------------3)--dataset {0} ---------------'.format(dataset))
                    train, train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)

                for tech in list_tech:
                    print('\n----------------{0}---------------'.format(tech))
                    for algo in list_algo:
                        print('\n******{0}**********'.format(algo))

                        classifier_machine_learning = classifier.Classifier(train=train, train_labels=train_labels, test=test, test_labels=test_labels, tech=tech, algo=algo, dataset=dataset)

                        # Entrainez votre classifieur

                        classifier_machine_learning.train(train, train_labels)

                        classifier_machine_learning.evaluate(test, test_labels)

                        time_after_excecution = datetime.now()
                        print("\nTime after execution = ", time_after_excecution)
                        print("\n Total execution time = ", time_after_excecution - time_before_excecution)



# Initialisez vos paramètres





# Initialisez/instanciez vos classifieurs avec leurs paramètres





# Charger/lire les datasets




# Entrainez votre classifieur


"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""




# Tester votre classifieur



"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""






