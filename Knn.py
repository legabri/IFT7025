from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn import neighbors, metrics
import numpy as np
import math

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import helper_functions

class Knn_Scikit:

    def __init__(self, **kwargs):
        self.train = kwargs['train']
        self.train_labels = kwargs['train_labels']
        self.dataset_key = kwargs['dataset_key']
        self.result_cross_validation = None
        self.best_k_chosen = None
        self.train_with_dummy_vars = None
        self.train_unique_value_sex = None
        self.pipe = None

    def get_model(self):

        if self.dataset_key == 'abalone':
            index_categ_var = 0
            column_transf = make_column_transformer((OneHotEncoder(), [index_categ_var]), remainder='passthrough')

            self.train_with_dummy_vars = column_transf.fit_transform(self.train)
            self.best_k_chosen, self.result_cross_validation = self.cross_validation_abalone()

            knn_model_temp = KNeighborsClassifier(n_neighbors=self.best_k_chosen)
            self.pipe = make_pipeline(column_transf, knn_model_temp)
            knn_model = self.pipe

        else:
            self.best_k_chosen, self.result_cross_validation = self.cross_validation()
            knn_model = KNeighborsClassifier(n_neighbors=self.best_k_chosen)

        return knn_model

    def cross_validation_abalone(self):
        param_grid = {'n_neighbors': [2, 3, 5, 7, 9, 11, 13, 15]}
        score = 'accuracy'
        model_sel = model_selection.GridSearchCV(
            neighbors.KNeighborsClassifier(),
            param_grid,
            cv=10,
            scoring=score
        )

        list_result = []

        if self.dataset_key == 'abalone':
            model_sel.fit(self.train_with_dummy_vars, self.train_labels)
        else:
            model_sel.fit(self.train, self.train_labels)

        for mean, std, params in zip(
                model_sel.cv_results_['mean_test_score'],
                model_sel.cv_results_['std_test_score'],
                model_sel.cv_results_['params']
        ):
            list_result.append(("{} = {:.4f} (+/-{:.04f}) for {}".format(
                score,
                mean,
                std * 2,
                params
            )))

        return model_sel.best_params_['n_neighbors'], list_result

    def cross_validation(self):
        param_grid = {'n_neighbors': [2, 3, 5, 7, 9, 11, 13, 15]}
        score = 'accuracy'
        model_sel = model_selection.GridSearchCV(
            neighbors.KNeighborsClassifier(),
            param_grid,
            cv=10,
            scoring=score
        )

        list_result = []

        if self.dataset_key == 'abalone':
            model_sel.fit(self.train, self.train_labels)
        else:
            model_sel.fit(self.train, self.train_labels)

        for mean, std, params in zip(
                model_sel.cv_results_['mean_test_score'],
                model_sel.cv_results_['std_test_score'],
                model_sel.cv_results_['params']
        ):
            list_result.append(("{} = {:.4f} (+/-{:.04f}) for {}".format(
                score,
                mean,
                std * 2,
                params
            )))

        return model_sel.best_params_['n_neighbors'], list_result

    def print_report(self, y, labels_predicted):
        print("accuracy test :\n", metrics.accuracy_score(y, labels_predicted))
        print("\nclassification report :\n", metrics.classification_report(y, labels_predicted))
        print("confusion matrix test :\n", metrics.confusion_matrix(y, labels_predicted))
        print("\nResult cross validation : ")
        for ele in self.result_cross_validation:
            print(ele)

        print("\nBest K chosen : ", self.best_k_chosen)

class Knn_Immplementation:

    def __init__(self, **kwargs):
        self.train = kwargs['train']
        self.train_labels = kwargs['train_labels']
        self.dataset_key = kwargs['dataset_key']

        self.result_cross_validation = None
        self.best_k_chosen = None

        self.train_unique_value_sex = None
        self.train_with_dummy_vars = None

    def cross_validate_model(self):
        if self.dataset_key == 'abalone':
            self.train_unique_value_sex = self.get_unique_item(self.train)
            self.train_with_dummy_vars = self.convert_dataset_abalone_to_float(self.train)

        self.best_k_chosen, self.result_cross_validation = self.cross_validation()

        return self.best_k_chosen, self.result_cross_validation

    def cross_validation(self):
        param_grid = {'n_neighbors': [2, 3, 5, 7, 9, 11, 13, 15]}
        score = 'accuracy'
        model_sel = model_selection.GridSearchCV(
            neighbors.KNeighborsClassifier(),
            param_grid,
            cv=10,
            scoring=score
        )

        list_result = []

        if self.dataset_key == 'abalone':
            model_sel.fit(self.train_with_dummy_vars, self.train_labels)
        else:
            model_sel.fit(self.train, self.train_labels)

        for mean, std, params in zip(
                model_sel.cv_results_['mean_test_score'],
                model_sel.cv_results_['std_test_score'],
                model_sel.cv_results_['params']
        ):
            list_result.append(("{} = {:.4f} (+/-{:.04f}) for {}".format(
                score,
                mean,
                std * 2,
                params
            )))

        return model_sel.best_params_['n_neighbors'], list_result

    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        return_value = ''
        list_predict = []
        index = 0

        if self.dataset_key == 'iris':
            for ele in self.train:
                distance = self.euclidean_distance_iris(ele, x)
                list_predict.append((distance, self.train_labels[index]))

                index = index + 1

            list_predict.sort(key=lambda tup: tup[0])

            list_k_neighbours = list_predict[0:self.best_k_chosen]

            list_labels = [y for (x, y) in list_k_neighbours]

            count_setosa, count_versicolor, count_virginica = helper_functions.sum_cal_iris(list_labels)

            if (count_setosa >= count_versicolor) and (count_setosa >= count_virginica):
                return_value = 'Iris-setosa'

            elif (count_versicolor >= count_setosa) and (count_versicolor >= count_virginica):
                return_value = 'Iris-versicolor'
            else:
                return_value = 'Iris-virginica'

        elif self.dataset_key == 'wine':
            for ele in self.train:
                distance = self.euclidean_distance_wine(ele, x)
                list_predict.append((distance, self.train_labels[index]))

                index = index + 1

            list_predict.sort(key=lambda tup: tup[0])

            list_k_neighbours = list_predict[0:self.best_k_chosen]

            list_labels = [y for (x, y) in list_k_neighbours]

            count_0, count_1 = helper_functions.sum_cal_wine(list_labels)

            if count_0 >= count_1:
                return_value = '0'
            else:
                return_value = '1'
        elif self.dataset_key == 'abalone':
            for ele in self.train:
                distance = self.euclidean_distance_abalone(ele, x)
                list_predict.append((distance, self.train_labels[index]))

                index = index + 1

            list_predict.sort(key=lambda tup: tup[0])

            list_k_neighbours = list_predict[0:self.best_k_chosen]

            list_labels = [y for (x, y) in list_k_neighbours]

            count_0 = list_labels.count('0.0')
            count_1 = list_labels.count('1.0')
            count_2 = list_labels.count('2.0')

            if (count_0 >= count_1) and (count_0 >= count_2):
                return_value = '0.0'
            elif (count_1 >= count_0) and (count_1 >= count_2):
                return_value = '1.0'
            else:
                return_value = '2.0'

        return return_value

    def euclidean_distance_iris(self, firstlist, secondlist):

        distance = math.sqrt(
            math.pow((firstlist[0] - secondlist[0]), 2) + math.pow((firstlist[1] - secondlist[1]), 2) + math.pow(
                (firstlist[2] - secondlist[2]), 2) + math.pow((firstlist[3] - secondlist[3]), 2))

        return distance

    def euclidean_distance_wine(self, firstlist, secondlist):

        distance = math.sqrt(
            math.pow((firstlist[0] - secondlist[0]), 2) + math.pow((firstlist[1] - secondlist[1]), 2) +
            math.pow((firstlist[2] - secondlist[2]), 2) + math.pow((firstlist[3] - secondlist[3]), 2) +
            math.pow((firstlist[4] - secondlist[4]), 2) + math.pow((firstlist[5] - secondlist[5]), 2) +
            math.pow((firstlist[6] - secondlist[6]), 2) + math.pow((firstlist[7] - secondlist[7]), 2) +
            math.pow((firstlist[8] - secondlist[8]), 2) + math.pow((firstlist[9] - secondlist[9]), 2) +
            math.pow((firstlist[10] - secondlist[10]), 2)
        )

        return distance

    def euclidean_distance_abalone(self, firstlist, secondlist):
        letter_first_vector_matrix = self.encode_text_label(firstlist[0])
        letter_second_vector_matrix = self.encode_text_label(secondlist[0])

        distance = math.sqrt(math.pow((letter_first_vector_matrix[0] - letter_second_vector_matrix[0]), 2) +
                             math.pow((letter_first_vector_matrix[1] - letter_second_vector_matrix[1]), 2) +
                             math.pow((letter_first_vector_matrix[2] - letter_second_vector_matrix[2]), 2) +
                             math.pow((float(firstlist[1]) - float(secondlist[1])), 2) + math.pow(
            (float(firstlist[2]) - float(secondlist[2])), 2) +
                             math.pow((float(firstlist[3]) - float(secondlist[3])), 2) + math.pow(
            (float(firstlist[4]) - float(secondlist[4])), 2) +
                             math.pow((float(firstlist[5]) - float(secondlist[5])), 2) + math.pow(
            (float(firstlist[6]) - float(secondlist[6])), 2) +
                             math.pow((float(firstlist[7]) - float(secondlist[7])), 2))

        return distance

    def encode_text_label(self, letter):
        matrix = [0] * len(self.train_unique_value_sex)
        x = self.train_unique_value_sex[letter]
        matrix[x] = 1

        return matrix

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
        labels_predicted = []
        index = 0

        if self.dataset_key == 'iris':
            count_false_setosa = 0
            count_false_versicolor = 0
            count_false_virginica = 0

            count_true_setosa = 0
            count_true_versicolor = 0
            count_true_virginica = 0

            for ele in X:
                label_predicted = self.predict(ele)
                labels_predicted.append(label_predicted)
                label_flower = y[index]

                if label_predicted != label_flower:
                    if label_predicted == 'Iris-setosa':
                        count_false_setosa = count_false_setosa + 1

                    elif label_predicted == 'Iris-versicolor':
                        count_false_versicolor = count_false_versicolor + 1

                    elif label_predicted == 'Iris-virginica':
                        count_false_virginica = count_false_virginica + 1
                else:
                    if label_predicted == 'Iris-setosa':
                        count_true_setosa = count_true_setosa + 1

                    elif label_predicted == 'Iris-versicolor':
                        count_true_versicolor = count_true_versicolor + 1

                    elif label_predicted == 'Iris-virginica':
                        count_true_virginica = count_true_virginica + 1

                index = index + 1

        elif self.dataset_key == 'wine':
            count_false_0 = 0
            count_false_1 = 0

            count_true_0 = 0
            count_true_1 = 0

            for ele in X:
                label_predicted = self.predict(ele)
                labels_predicted.append(label_predicted)
                label_flower = y[index]

                if label_predicted != label_flower:
                    if label_predicted == '0':
                        count_false_0 = count_false_0 + 1

                    elif label_predicted == '1':
                        count_false_1 = count_false_1 + 1
                else:
                    if label_predicted == '0':
                        count_true_0 = count_true_0 + 1

                    elif label_predicted == '1':
                        count_true_1 = count_true_1 + 1
                index = index + 1

        elif self.dataset_key == 'abalone':
            count_false_0 = 0
            count_false_1 = 0
            count_false_2 = 0

            count_true_0 = 0
            count_true_1 = 0
            count_true_2 = 0

            for ele in X:
                label_predicted = self.predict(ele)
                labels_predicted.append(label_predicted)
                label_flower = y[index]

                if label_predicted != label_flower:
                    if label_predicted == '0.0':
                        count_false_0 = count_false_0 + 1

                    elif label_predicted == '1.0':
                        count_false_1 = count_false_1 + 1

                    elif label_predicted == '2.0':
                        count_false_2 = count_false_2 + 1
                else:
                    if label_predicted == '0.0':
                        count_true_0 = count_true_0 + 1

                    elif label_predicted == '1.0':
                        count_true_1 = count_true_1 + 1

                    elif label_predicted == '2.0':
                        count_true_2 = count_true_2 + 1

                index = index + 1

        return labels_predicted

    def print_report(self, y, labels_predicted):
        confusion_matrix = None
        accuracy = None
        classification_report = None

        if self.dataset_key == 'iris':
            confusion_matrix, accuracy, classification_report = helper_functions.compute_report_iris(y, labels_predicted)
        elif self.dataset_key == 'wine':
            confusion_matrix, accuracy, classification_report = helper_functions.compute_report_wine(y, labels_predicted)
        elif self.dataset_key == 'abalone':
            confusion_matrix, accuracy, classification_report = helper_functions.compute_report_abalone(y, labels_predicted)

        print("\nAccuracy: ", accuracy)
        print("\n")
        print("Classification report:\n", classification_report)
        print("\n")
        print("Confusion matrix:\n", confusion_matrix)

        print("\nTotal count train data : ", len(self.train_labels))
        print("\nTotal count test data : ", len(y))

        print("\nResult cross validation : ")
        for ele in self.result_cross_validation:
            print(ele)

        print("\nBest K chosen : ", self.best_k_chosen)

    def convert_dataset_abalone_to_float(self, vector):
        list_result = []
        for ele in vector:
            list_attr = []
            matrix = self.encode_text_label_abalone(ele[0])
            for el in matrix:
                list_attr.append(el)

            list_temp = ele[1:]
            for el in list_temp:
                list_attr.append(el)

            list_result.append(list_attr)

        list_result_mat = np.asarray(list_result)

        return list_result_mat

    def encode_text_label_abalone(self, letter):
        matrix = [0.0] * len(self.train_unique_value_sex)
        x = self.train_unique_value_sex[letter]
        matrix[x] = 1.0

        return matrix

    def get_unique_item(self, vector):
        list_items = [i[0] for i in vector]
        unique = list(set(list_items))
        unique.sort()
        lookup = dict()

        for i, value in enumerate(unique):
            lookup[value] = i

        return lookup
