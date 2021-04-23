import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
       
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.

    train, train_labels, test, test_labels = load_datasets(f, train_ratio, True)

    return (train, train_labels, test, test_labels)
	
def load_datasets(f, train_ratio, convertalltofloat):
    lines = f.read().splitlines()
    random.shuffle(lines)

    length_file = len(lines)
    max_length_train = int(length_file * train_ratio)
    train_data = lines[0:max_length_train]
    test_data = lines[max_length_train:length_file+1]

    train_list, train_labels_list = zip(*[tuple(s.rsplit(',', 1)) for s in train_data])
    test_list, test_labels_list = zip(*[tuple(s.rsplit(',', 1)) for s in test_data])

    train, train_labels, test, test_labels = convert_to_numpy_matrix(train_list, train_labels_list, test_list, test_labels_list, convertalltofloat)

    return (train, train_labels, test, test_labels)
	
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

	
    # TODO : le code ici pour lire le dataset
    
	
	# La fonction doit retourner 4 structures de données de type Numpy.

    train, train_labels, test, test_labels = load_datasets(f, train_ratio, True)

    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.

    train, train_labels, test, test_labels = load_datasets(f, train_ratio, False)

    return (train, train_labels, test, test_labels)

def convert_to_numpy_matrix(train_list, train_labels_list, test_list, test_labels_list, convertalltofloat):
    if convertalltofloat:
        train = [[float(i) for i in ele.split(',')] for ele in train_list]
        test = [[float(i) for i in ele.split(',')] for ele in test_list]
    else:
        list_result = []
        for ele in train_list:
            ele_items = ele.split(',')
            list_attr = []
            for i in range(len(ele_items)):
                if i == 0:
                    list_attr.append(ele_items[i])
                else:
                    list_attr.append(float(ele_items[i]))
            list_result.append(list_attr)
        train = list_result

        list_result = []
        for ele in test_list:
            ele_items = ele.split(',')
            list_attr = []
            for i in range(len(ele_items)):
                if i == 0:
                    list_attr.append(ele_items[i])
                else:
                    list_attr.append(float(ele_items[i]))
            list_result.append(list_attr)
        test = list_result

    train_mat = np.asarray(train, dtype=object)
    train_labels_mat = np.asarray(train_labels_list)

    test_mat = np.asarray(test, dtype=object)
    test_labels_mat = np.asarray(test_labels_list)

    return (train_mat, train_labels_mat, test_mat, test_labels_mat)