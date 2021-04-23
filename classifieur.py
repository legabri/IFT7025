"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import Knn as knnfile
import NaiveBayes as nbfile


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins
import helper_functions


class Classifier: #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""

		self.train_ = kwargs['train']
		self.train_labels = kwargs['train_labels']
		self.test = kwargs['test']
		self.test_labels = kwargs['test_labels']
		self.test_labels = kwargs['test_labels']
		self.tech = kwargs['tech']
		self.algo = kwargs['algo']
		self.dataset_key = kwargs['dataset']

		self.algo_model = None
		self.result_cross_validation = None
		self.best_k_chosen = None

		self.train_predicted = None
		self.test_predicted = None

		self.train_with_dummy_vars = None
		self.best_k_chosen = None

		self.algo_file = None
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""

		self.train_ = train
		self.train_labels = train_labels

		if self.dataset_key == 'abalone':
			if self.tech == 'knn':
				if self.algo == 'implementation':
					self.algo_file = knnfile.Knn_Immplementation(train=self.train_, train_labels=self.train_labels,
																 dataset_key=self.dataset_key)
					self.best_k_chosen, self.result_cross_validation = self.algo_file.cross_validate_model()

				elif self.algo == 'scikit':
					self.algo_file = knnfile.Knn_Scikit(train=self.train_, train_labels=self.train_labels,
														dataset_key=self.dataset_key)
					self.algo_model = self.algo_file.get_model()
					self.algo_model.fit(self.train_, self.train_labels)

			elif self.tech == 'nb':
				if self.algo == 'implementation':
					self.algo_file = nbfile.NB_Immplementation(train=self.train_, train_labels=self.train_labels,
															   dataset_key=self.dataset_key)
					self.algo_file.compute_mean_and_stddev()

				elif self.algo == 'scikit':
					self.algo_file = nbfile.NB_Scikit(train=self.train_, train_labels=self.train_labels,
													  dataset_key=self.dataset_key)
					self.algo_model = self.algo_file.get_model()
					self.algo_model.fit(self.train_, self.train_labels)

		else:
			if self.tech == 'knn':
				if self.algo == 'implementation':
					self.algo_file = knnfile.Knn_Immplementation(train=self.train_, train_labels=self.train_labels, dataset_key=self.dataset_key)
					self.best_k_chosen, self.result_cross_validation = self.algo_file.cross_validate_model()

				elif self.algo == 'scikit':
					self.algo_file = knnfile.Knn_Scikit(train=self.train_, train_labels=self.train_labels, dataset_key=self.dataset_key)
					self.algo_model = self.algo_file.get_model()
					self.algo_model.fit(self.train_, self.train_labels)

			elif self.tech == 'nb':
				if self.algo == 'implementation':
					self.algo_file = nbfile.NB_Immplementation(train=self.train_, train_labels=self.train_labels, dataset_key=self.dataset_key)
					self.algo_file.compute_mean_and_stddev()

				elif self.algo == 'scikit':
					self.algo_file = nbfile.NB_Scikit(train=self.train_, train_labels=self.train_labels, dataset_key=self.dataset_key)
					self.algo_model = self.algo_file.get_model()
					self.algo_model.fit(self.train_, self.train_labels)

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""

		if self.tech == 'knn':
			if self.algo == 'implementation':
				return self.algo_file.predict(x)
			elif self.algo == 'scikit':
				return self.algo_model.predict(x)

		elif self.tech == 'nb':
			if self.algo == 'implementation':
				return self.algo_file.predict(x)

			elif self.algo == 'scikit':
				return self.algo_model.predict(x)

	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""

		if self.tech == 'knn':
			if self.algo == 'implementation':
				labels_predicted = self.algo_file.evaluate(X, y)
				self.algo_file.print_report(y, labels_predicted)
			elif self.algo == 'scikit':
				labels_predicted = self.predict(X)
				self.algo_file.print_report(y, labels_predicted)

		elif self.tech == 'nb':
			if self.algo == 'implementation':
				labels_predicted = self.algo_file.evaluate(X, y)
				self.algo_file.print_report(y, labels_predicted)
			elif self.algo == 'scikit':
				labels_predicted = self.predict(X)
				self.algo_file.print_report(y, labels_predicted)

	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.