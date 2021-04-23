from sklearn.compose import make_column_transformer
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import neighbors, metrics
import numpy as np
import math

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import helper_functions

class NB_Scikit:

    def __init__(self, **kwargs):
        self.train = kwargs['train']
        self.train_labels = kwargs['train_labels']
        self.dataset_key = kwargs['dataset_key']
        self.nb_model = None

        self.train_with_dummy_vars = None
        self.pipe = None


    def get_model(self):

        if self.dataset_key == 'abalone':
            index_categ_var = 0

            column_transform = make_column_transformer((OneHotEncoder(), [index_categ_var]), remainder='passthrough')

            self.nb_model = GaussianNB()
            self.pipe = make_pipeline(column_transform, self.nb_model)
            self.nb_model = self.pipe

        else:
            self.nb_model = GaussianNB()

        return self.nb_model

    def print_report(self, y, labels_predicted):
        print("accuracy test :\n", metrics.accuracy_score(y, labels_predicted))
        print("\nclassification report :\n", metrics.classification_report(y, labels_predicted))
        print("confusion matrix test :\n", metrics.confusion_matrix(y, labels_predicted))
        print("\nResult cross validation : ")

class NB_Immplementation:

    def __init__(self, **kwargs):
        self.train = kwargs['train']
        self.train_labels = kwargs['train_labels']
        self.dataset_key = kwargs['dataset_key']

        #data set iris
        self.avg_sepal_length_setosa = 0.0
        self.avg_sepal_width_setosa = 0.0
        self.avg_petal_length_setosa = 0.0
        self.avg_petal_width_setosa = 0.0

        self.avg_sepal_length_versicolor = 0.0
        self.avg_sepal_width_versicolor = 0.0
        self.avg_petal_length_versicolor = 0.0
        self.avg_petal_width_versicolor = 0.0

        self.avg_sepal_length_virginica = 0.0
        self.avg_sepal_width_virginica = 0.0
        self.avg_petal_length_virginica = 0.0
        self.avg_petal_width_virginica = 0.0

        self.std_sepal_length_setosa = 0.0
        self.std_sepal_width_setosa = 0.0
        self.std_petal_length_setosa = 0.0
        self.std_petal_width_setosa = 0.0

        self.std_sepal_length_versicolor = 0.0
        self.std_sepal_width_versicolor = 0.0
        self.std_petal_length_versicolor = 0.0
        self.std_petal_width_versicolor = 0.0

        self.std_sepal_length_virginica = 0.0
        self.std_sepal_width_virginica = 0.0
        self.std_petal_length_virginica = 0.0
        self.std_petal_width_virginica = 0.0

        # data set Wine
        self.avg_fixed_acid_0 = 0.0
        self.avg_vol_acid_0 = 0.0
        self.avg_citric_acid_0 = 0.0
        self.avg_resid_sugar_0 = 0.0
        self.avg_chlorid_0 = 0.0
        self.avg_free_sulf_dioxid_0 = 0.0
        self.avg_total_sulf_dioxid_0 = 0.0
        self.avg_density_0 = 0.0
        self.avg_ph_0 = 0.0
        self.avg_sulfate_0 = 0.0
        self.avg_alcohol_0 = 0.0

        self.avg_fixed_acid_1 = 0.0
        self.avg_vol_acid_1 = 0.0
        self.avg_citric_acid_1 = 0.0
        self.avg_resid_sugar_1 = 0.0
        self.avg_chlorid_1 = 0.0
        self.avg_free_sulf_dioxid_1 = 0.0
        self.avg_total_sulf_dioxid_1 = 0.0
        self.avg_density_1 = 0.0
        self.avg_ph_1 = 0.0
        self.avg_sulfate_1 = 0.0
        self.avg_alcohol_1 = 0.0

        self.std_fixed_acid_0 = 0.0
        self.std_vol_acid_0 = 0.0
        self.std_citric_acid_0 = 0.0
        self.std_resid_sugar_0 = 0.0
        self.std_chlorid_0 = 0.0
        self.std_free_sulf_dioxid_0 = 0.0
        self.std_total_sulf_dioxid_0 = 0.0
        self.std_density_0 = 0.0
        self.std_ph_0 = 0.0
        self.std_sulfate_0 = 0.0
        self.std_alcohol_0 = 0.0

        self.std_fixed_acid_1 = 0.0
        self.std_vol_acid_1 = 0.0
        self.std_citric_acid_1 = 0.0
        self.std_resid_sugar_1 = 0.0
        self.std_chlorid_1 = 0.0
        self.std_free_sulf_dioxid_1 = 0.0
        self.std_total_sulf_dioxid_1 = 0.0
        self.std_density_1 = 0.0
        self.std_ph_1 = 0.0
        self.std_sulfate_1 = 0.0
        self.std_alcohol_1 = 0.0

        self.train_unique_value_sex = None
        self.vector_categ_var = None
        self.categ_vars_probas = None

        self.avg_sex_0 = 0.0
        self.avg_length_0 = 0.0
        self.avg_diameter_0 = 0.0
        self.avg_height_0 = 0.0
        self.avg_whol_weight_0 = 0.0
        self.avg_shuc_weight_0 = 0.0
        self.avg_visce_weight_0 = 0.0
        self.avg_shell_weight_0 = 0.0

        self.avg_sex_1 = 0.0
        self.avg_length_1 = 0.0
        self.avg_diameter_1 = 0.0
        self.avg_height_1 = 0.0
        self.avg_whol_weight_1 = 0.0
        self.avg_shuc_weight_1 = 0.0
        self.avg_visce_weight_1 = 0.0
        self.avg_shell_weight_1 = 0.0

        self.avg_sex_2 = 0.0
        self.avg_length_2 = 0.0
        self.avg_diameter_2 = 0.0
        self.avg_height_2 = 0.0
        self.avg_whol_weight_2 = 0.0
        self.avg_shuc_weight_2 = 0.0
        self.avg_visce_weight_2 = 0.0
        self.avg_shell_weight_2 = 0.0

        self.std_sex_0 = 0.0
        self.std_length_0 = 0.0
        self.std_diameter_0 = 0.0
        self.std_height_0 = 0.0
        self.std_whol_weight_0 = 0.0
        self.std_shuc_weight_0 = 0.0
        self.std_visce_weight_0 = 0.0
        self.std_shell_weight_0 = 0.0

        self.std_sex_1 = 0.0
        self.std_length_1 = 0.0
        self.std_diameter_1 = 0.0
        self.std_height_1 = 0.0
        self.std_whol_weight_1 = 0.0
        self.std_shuc_weight_1 = 0.0
        self.std_visce_weight_1 = 0.0
        self.std_shell_weight_1 = 0.0

        self.std_sex_2 = 0.0
        self.std_length_2 = 0.0
        self.std_diameter_2 = 0.0
        self.std_height_2 = 0.0
        self.std_whol_weight_2 = 0.0
        self.std_shuc_weight_2 = 0.0
        self.std_visce_weight_2 = 0.0
        self.std_shell_weight_2 = 0.0


    def compute_mean_and_stddev(self):
        if self.dataset_key == 'iris':
            self.avg_sepal_length_setosa, self.avg_sepal_length_versicolor, self.avg_sepal_length_virginica = self.mean_cal_iris(
                0)
            self.avg_sepal_width_setosa, self.avg_sepal_width_versicolor, self.avg_sepal_width_virginica = self.mean_cal_iris(1)
            self.avg_petal_length_setosa, self.avg_petal_length_versicolor, self.avg_petal_length_virginica = self.mean_cal_iris(
                2)
            self.avg_petal_width_setosa, self.avg_petal_width_versicolor, self.avg_petal_width_virginica = self.mean_cal_iris(3)

            self.std_sepal_length_setosa, self.std_sepal_length_versicolor, self.std_sepal_length_virginica = self.stddev_cal_iris(
                0, self.avg_sepal_length_setosa, self.avg_sepal_length_versicolor, self.avg_sepal_length_virginica)
            self.std_sepal_width_setosa, self.std_sepal_width_versicolor, self.std_sepal_width_virginica = self.stddev_cal_iris(
                1, self.avg_sepal_width_setosa, self.avg_sepal_width_versicolor, self.avg_sepal_width_virginica)
            self.std_petal_length_setosa, self.std_petal_length_versicolor, self.std_petal_length_virginica = self.stddev_cal_iris(
                2, self.avg_petal_length_setosa, self.avg_petal_length_versicolor, self.avg_petal_length_virginica)
            self.std_petal_width_setosa, self.std_petal_width_versicolor, self.std_petal_width_virginica = self.stddev_cal_iris(
                3, self.avg_petal_width_setosa, self.avg_petal_width_versicolor, self.avg_petal_width_virginica)
        elif self.dataset_key == 'wine':
            self.avg_fixed_acid_0, self.avg_fixed_acid_1 = self.mean_cal_wine(0)
            self.avg_vol_acid_0, self.avg_vol_acid_1 = self.mean_cal_wine(1)
            self.avg_citric_acid_0, self.avg_citric_acid_1 = self.mean_cal_wine(2)
            self.avg_resid_sugar_0, self.avg_resid_sugar_1 = self.mean_cal_wine(3)
            self.avg_chlorid_0, self.avg_chlorid_1 = self.mean_cal_wine(4)
            self.avg_free_sulf_dioxid_0, self.avg_free_sulf_dioxid_1 = self.mean_cal_wine(5)
            self.avg_total_sulf_dioxid_0, self.avg_total_sulf_dioxid_1 = self.mean_cal_wine(6)
            self.avg_density_0, self.avg_density_1 = self.mean_cal_wine(7)
            self.avg_ph_0, self.avg_ph_1 = self.mean_cal_wine(8)
            self.avg_sulfate_0, self.avg_sulfate_1 = self.mean_cal_wine(9)
            self.avg_alcohol_0, self.avg_alcohol_1 = self.mean_cal_wine(10)

            self.std_fixed_acid_0, self.std_fixed_acid_1 = self.stddev_cal_wine(0, self.avg_fixed_acid_0,
                                                                                self.avg_fixed_acid_1)
            self.std_vol_acid_0, self.std_vol_acid_1 = self.stddev_cal_wine(1, self.avg_vol_acid_0, self.avg_vol_acid_1)
            self.std_citric_acid_0, self.std_citric_acid_1 = self.stddev_cal_wine(2, self.avg_citric_acid_0,
                                                                                  self.avg_citric_acid_1)
            self.std_resid_sugar_0, self.std_resid_sugar_1 = self.stddev_cal_wine(3, self.avg_resid_sugar_0,
                                                                                  self.avg_resid_sugar_1)
            self.std_chlorid_0, self.std_chlorid_1 = self.stddev_cal_wine(4, self.avg_chlorid_0, self.avg_chlorid_1)
            self.std_free_sulf_dioxid_0, self.std_free_sulf_dioxid_1 = self.stddev_cal_wine(5, self.avg_free_sulf_dioxid_0,
                                                                                            self.avg_free_sulf_dioxid_1)
            self.std_total_sulf_dioxid_0, self.std_total_sulf_dioxid_1 = self.stddev_cal_wine(6,
                                                                                              self.avg_total_sulf_dioxid_0,
                                                                                              self.avg_total_sulf_dioxid_1)
            self.std_density_0, self.std_density_1 = self.stddev_cal_wine(7, self.avg_density_0, self.avg_density_1)
            self.std_ph_0, self.std_ph_1 = self.stddev_cal_wine(8, self.avg_ph_0, self.avg_ph_1)
            self.std_sulfate_0, self.std_sulfate_1 = self.stddev_cal_wine(9, self.avg_sulfate_0, self.avg_sulfate_1)
            self.std_alcohol_0, self.std_alcohol_1 = self.stddev_cal_wine(10, self.avg_alcohol_0, self.avg_alcohol_1)

        elif self.dataset_key == 'abalone':
            self.train_unique_value_sex = helper_functions.get_unique_item(self.train)
            self.vector_categ_var = self.train[:, 0]
            self.categ_vars_probas = self.categ_vars_cal()

            self.avg_length_0, self.avg_length_1, self.avg_length_2 = self.mean_cal_abalone(1)
            self.avg_diameter_0, self.avg_diameter_1, self.avg_diameter_2 = self.mean_cal_abalone(2)
            self.avg_height_0, self.avg_height_1, self.avg_height_2 = self.mean_cal_abalone(3)
            self.avg_whol_weight_0, self.avg_whol_weight_1, self.avg_whol_weight_2 = self.mean_cal_abalone(4)
            self.avg_shuc_weight_0, self.avg_shuc_weight_1, self.avg_shuc_weight_2 = self.mean_cal_abalone(5)
            self.avg_visce_weight_0, self.avg_visce_weight_1, self.avg_visce_weight_2 = self.mean_cal_abalone(6)
            self.avg_shell_weight_0, self.avg_shell_weight_1, self.avg_shell_weight_2 = self.mean_cal_abalone(7)

            self.std_length_0, self.std_length_1, self.std_length_2 = self.stddev_cal_abalone(1, self.avg_length_0,
                                                                                              self.avg_length_1,
                                                                                              self.avg_length_2)
            self.std_diameter_0, self.std_diameter_1, self.std_diameter_2 = self.stddev_cal_abalone(2, self.avg_diameter_0,
                                                                                                    self.avg_diameter_1,
                                                                                                    self.avg_diameter_2)
            self.std_height_0, self.std_height_1, self.std_height_2 = self.stddev_cal_abalone(3, self.avg_height_0,
                                                                                              self.avg_height_1,
                                                                                              self.avg_height_2)
            self.std_whol_weight_0, self.std_whol_weight_1, self.std_whol_weight_2 = self.stddev_cal_abalone(4,
                                                                                                             self.avg_whol_weight_0,
                                                                                                             self.avg_whol_weight_1,
                                                                                                             self.avg_whol_weight_2)
            self.std_shuc_weight_0, self.std_shuc_weight_1, self.std_shuc_weight_2 = self.stddev_cal_abalone(5,
                                                                                                             self.avg_shuc_weight_0,
                                                                                                             self.avg_shuc_weight_1,
                                                                                                             self.avg_shuc_weight_2)
            self.std_visce_weight_0, self.std_visce_weight_1, self.std_visce_weight_2 = self.stddev_cal_abalone(6,
                                                                                                                self.avg_visce_weight_0,
                                                                                                                self.avg_visce_weight_1,
                                                                                                                self.avg_visce_weight_2)
            self.std_shell_weight_0, self.std_shell_weight_1, self.std_shell_weight_2 = self.stddev_cal_abalone(7,
                                                                                                                self.avg_shell_weight_0,
                                                                                                                self.avg_shell_weight_1,
                                                                                                                self.avg_shell_weight_2)

    def mean_cal_iris(self, index_attr):

        index_train_labels = 0

        sum_setosa = 0.0
        sum_versicolor = 0.0
        sum_virginica = 0.0

        count_label_setosa = 0
        count_label_versicolor = 0
        count_label_virginica = 0

        for ele in self.train:
            label = self.train_labels[index_train_labels]
            if label == 'Iris-setosa':
                sum_setosa = sum_setosa + ele[index_attr]

                count_label_setosa = count_label_setosa + 1
            elif label == 'Iris-versicolor':
                sum_versicolor = sum_versicolor + ele[index_attr]

                count_label_versicolor = count_label_versicolor + 1
            elif label == 'Iris-virginica':
                sum_virginica = sum_virginica + ele[index_attr]
                count_label_virginica = count_label_virginica + 1

            index_train_labels = index_train_labels + 1

        avg_setosa = sum_setosa / count_label_setosa
        avg_versicolor = sum_versicolor / count_label_versicolor
        avg_virginica = sum_virginica / count_label_virginica

        return avg_setosa, avg_versicolor, avg_virginica

    def stddev_cal_iris(self, index_attr, avg_setosa, avg_versicolor, avg_virginica):
        index_train_labels = 0

        sum_setosa = 0.0
        sum_versicolor = 0.0
        sum_virginica = 0.0

        count_label_setosa = 0
        count_label_versicolor = 0
        count_label_virginica = 0

        for ele in self.train:
            label = self.train_labels[index_train_labels]
            if label == 'Iris-setosa':
                sum_setosa = sum_setosa + math.pow((ele[index_attr] - avg_setosa), 2)
                count_label_setosa = count_label_setosa + 1

            elif label == 'Iris-versicolor':
                sum_versicolor = sum_versicolor + math.pow((ele[index_attr] - avg_versicolor), 2)
                count_label_versicolor = count_label_versicolor + 1

            elif label == 'Iris-virginica':
                sum_virginica = sum_virginica + math.pow((ele[index_attr] - avg_virginica), 2)
                count_label_virginica = count_label_virginica + 1

            index_train_labels = index_train_labels + 1

        std_setosa = sum_setosa / count_label_setosa - 1
        std_versicolor = sum_versicolor / count_label_versicolor - 1
        std_virginica = sum_virginica / count_label_virginica - 1

        return std_setosa, std_versicolor, std_virginica

    def mean_cal_wine(self, index_attr):

        index_train_labels = 0

        sum_0 = 0.0
        sum_1 = 0.0

        count_label_0 = 0
        count_label_1 = 0

        for ele in self.train:
            label = self.train_labels[index_train_labels]
            if label == '0':
                sum_0 = sum_0 + ele[index_attr]

                count_label_0 = count_label_0 + 1
            else:
                sum_1 = sum_1 + ele[index_attr]

                count_label_1 = count_label_1 + 1

            index_train_labels = index_train_labels + 1

        avg_0 = sum_0 / count_label_0
        avg_1 = sum_1 / count_label_1

        return avg_0, avg_1

    def stddev_cal_wine(self, index_attr, avg_0, avg_1):
        index_train_labels = 0

        sum_0 = 0.0
        sum_1 = 0.0

        count_label_0 = 0
        count_label_1 = 0

        for ele in self.train:
            label = self.train_labels[index_train_labels]
            if label == '0':
                sum_0 = sum_0 + math.pow((ele[index_attr] - avg_0), 2)
                count_label_0 = count_label_0 + 1

            else:
                sum_1 = sum_1 + math.pow((ele[index_attr] - avg_1), 2)
                count_label_1 = count_label_1 + 1

            index_train_labels = index_train_labels + 1

        std_0 = sum_0 / count_label_0 - 1
        std_1 = sum_1 / count_label_1 - 1

        return std_0, std_1

    def categ_vars_cal(self):

        index_train_labels = 0

        count_label_F_0 = 0
        count_label_F_1 = 0
        count_label_F_2 = 0
        count_label_I_0 = 0
        count_label_I_1 = 0
        count_label_I_2 = 0
        count_label_M_0 = 0
        count_label_M_1 = 0
        count_label_M_2 = 0

        for ele in self.vector_categ_var:
            if ele == 'F':
                label = self.train_labels[index_train_labels]
                if label == '0.0':
                    count_label_F_0 = count_label_F_0 + 1
                elif label == '1.0':
                    count_label_F_1 = count_label_F_1 + 1
                elif label == '2.0':
                    count_label_F_2 = count_label_F_2 + 1

            elif ele == 'I':
                label = self.train_labels[index_train_labels]
                if label == '0.0':
                    count_label_I_0 = count_label_I_0 + 1
                elif label == '1.0':
                    count_label_I_1 = count_label_I_1 + 1
                elif label == '2.0':
                    count_label_I_2 = count_label_I_2 + 1

            elif ele == 'M':
                label = self.train_labels[index_train_labels]
                if label == '0.0':
                    count_label_M_0 = count_label_M_0 + 1
                elif label == '1.0':
                    count_label_M_1 = count_label_M_1 + 1
                elif label == '2.0':
                    count_label_M_2 = count_label_M_2 + 1

            index_train_labels = index_train_labels + 1

        count_label_0, count_label_1, count_label_2 = helper_functions.sum_cal_abalone(self.train_labels)
        length_unique_var = len(self.train_unique_value_sex)

        p_label_F_0 = (count_label_F_0 + 1) / (count_label_0 + length_unique_var)
        p_label_F_1 = (count_label_F_1 + 1) / (count_label_1 + length_unique_var)
        p_label_F_2 = (count_label_F_2 + 1) / (count_label_2 + length_unique_var)

        p_label_I_0 = (count_label_I_0 + 1) / (count_label_0 + length_unique_var)
        p_label_I_1 = (count_label_I_1 + 1) / (count_label_1 + length_unique_var)
        p_label_I_2 = (count_label_I_2 + 1) / (count_label_2 + length_unique_var)

        p_label_M_0 = (count_label_M_0 + 1) / (count_label_0 + length_unique_var)
        p_label_M_1 = (count_label_M_1 + 1) / (count_label_1 + length_unique_var)
        p_label_M_2 = (count_label_M_2 + 1) / (count_label_2 + length_unique_var)

        result = dict()
        result['F'] = [p_label_F_0, p_label_F_1, p_label_F_2]
        result['I'] = [p_label_I_0, p_label_I_1, p_label_I_2]
        result['M'] = [p_label_M_0, p_label_M_1, p_label_M_2]

        return result

    def mean_cal_abalone(self, index_attr):

        index_train_labels = 0

        sum_0 = 0.0
        sum_1 = 0.0
        sum_2 = 0.0

        count_label_0 = 0
        count_label_1 = 0
        count_label_2 = 0

        for ele in self.train:
            label = self.train_labels[index_train_labels]
            if label == '0.0':
                sum_0 = sum_0 + ele[index_attr]
                count_label_0 = count_label_0 + 1

            elif label == '1.0':
                sum_1 = sum_1 + ele[index_attr]
                count_label_1 = count_label_1 + 1

            elif label == '2.0':
                sum_2 = sum_2 + ele[index_attr]
                count_label_2 = count_label_2 + 1

            index_train_labels = index_train_labels + 1

        avg_0 = sum_0 / count_label_0
        avg_1 = sum_1 / count_label_1
        avg_2 = sum_2 / count_label_2

        return avg_0, avg_1, avg_2

    def stddev_cal_abalone(self, index_attr, avg_0, avg_1, avg_2):
        index_train_labels = 0

        sum_0 = 0.0
        sum_1 = 0.0
        sum_2 = 0.0

        count_label_0 = 0
        count_label_1 = 0
        count_label_2 = 0

        for ele in self.train:
            label = self.train_labels[index_train_labels]
            if label == '0.0':
                sum_0 = sum_0 + math.pow((ele[index_attr] - avg_0), 2)
                count_label_0 = count_label_0 + 1

            elif label == '1.0':
                sum_1 = sum_1 + math.pow((ele[index_attr] - avg_1), 2)
                count_label_1 = count_label_1 + 1

            elif label == '2.0':
                sum_2 = sum_2 + math.pow((ele[index_attr] - avg_2), 2)
                count_label_2 = count_label_2 + 1

            index_train_labels = index_train_labels + 1

        std_0 = sum_0 / count_label_0 - 1
        std_1 = sum_1 / count_label_1 - 1
        std_2 = sum_2 / count_label_2 - 1

        return std_0, std_1, std_2

    def gaussian_cal(self, value_att, avg_att, std_dev_att):
        fisrt_expre_denom = math.sqrt(2 * math.pi * math.pow(std_dev_att, 2))
        fisrt_expre = 1 / fisrt_expre_denom

        second_expre_expo = (math.pow((value_att - avg_att), 2)) / (2 * (math.pow(std_dev_att, 2)))
        second_expre = math.exp(-second_expre_expo)

        gaussian_cal = fisrt_expre * second_expre

        return gaussian_cal

    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """

        return_value = ''

        if self.dataset_key == 'iris':

            gaussian_sepal_length_setosa = self.gaussian_cal(x[0], self.avg_sepal_length_setosa,
                                                             self.std_sepal_length_setosa)
            gaussian_sepal_width_setosa = self.gaussian_cal(x[1], self.avg_sepal_width_setosa, self.std_sepal_width_setosa)
            gaussian_petal_length_setosa = self.gaussian_cal(x[2], self.avg_petal_length_setosa,
                                                             self.std_petal_length_setosa)
            gaussian_petal_width_setosa = self.gaussian_cal(x[3], self.avg_petal_width_setosa, self.std_petal_width_setosa)

            gaussian_sepal_length_versicolor = self.gaussian_cal(x[0], self.avg_sepal_length_versicolor,
                                                                 self.std_sepal_length_versicolor)
            gaussian_sepal_width_versicolor = self.gaussian_cal(x[1], self.avg_sepal_width_versicolor,
                                                                self.std_sepal_width_versicolor)
            gaussian_petal_length_versicolor = self.gaussian_cal(x[2], self.avg_petal_length_versicolor,
                                                                 self.std_petal_length_versicolor)
            gaussian_petal_width_versicolor = self.gaussian_cal(x[3], self.avg_petal_width_versicolor,
                                                                self.std_petal_width_versicolor)

            gaussian_sepal_length_virginica = self.gaussian_cal(x[0], self.avg_sepal_length_virginica,
                                                                self.std_sepal_length_virginica)
            gaussian_sepal_width_virginica = self.gaussian_cal(x[1], self.avg_sepal_width_virginica,
                                                               self.std_sepal_width_virginica)
            gaussian_petal_length_virginica = self.gaussian_cal(x[2], self.avg_petal_length_virginica,
                                                                self.std_petal_length_virginica)
            gaussian_petal_width_virginica = self.gaussian_cal(x[3], self.avg_petal_width_virginica,
                                                               self.std_petal_width_virginica)

            proba_setosa = gaussian_sepal_length_setosa * gaussian_sepal_width_setosa * gaussian_petal_length_setosa * gaussian_petal_width_setosa
            proba_versicolor = gaussian_sepal_length_versicolor * gaussian_sepal_width_versicolor * gaussian_petal_length_versicolor * gaussian_petal_width_versicolor
            proba_virginica = gaussian_sepal_length_virginica * gaussian_sepal_width_virginica * gaussian_petal_length_virginica * gaussian_petal_width_virginica

            if (proba_setosa >= proba_versicolor) and (proba_setosa >= proba_virginica):
                return_value = 'Iris-setosa'

            elif (proba_versicolor >= proba_setosa) and (proba_versicolor >= proba_virginica):
                return_value = 'Iris-versicolor'
            else:
                return_value = 'Iris-virginica'

        elif self.dataset_key == 'wine':
            gaussian_fixed_acid_0 = self.gaussian_cal(x[0], self.avg_fixed_acid_0, self.std_fixed_acid_0)
            gaussian_vol_acid_0 = self.gaussian_cal(x[1], self.avg_vol_acid_0, self.std_vol_acid_0)
            gaussian_citric_acid_0 = self.gaussian_cal(x[2], self.avg_citric_acid_0, self.std_citric_acid_0)
            gaussian_resid_sugar_0 = self.gaussian_cal(x[3], self.avg_resid_sugar_0, self.std_resid_sugar_0)
            gaussian_chlorid_0 = self.gaussian_cal(x[4], self.avg_chlorid_0, self.std_chlorid_0)
            gaussian_free_sulf_dioxid_0 = self.gaussian_cal(x[5], self.avg_free_sulf_dioxid_0,
                                                            self.std_free_sulf_dioxid_0)
            gaussian_total_sulf_dioxid_0 = self.gaussian_cal(x[6], self.avg_total_sulf_dioxid_0,
                                                             self.std_total_sulf_dioxid_0)
            gaussian_density_0 = self.gaussian_cal(x[7], self.avg_density_0, self.std_density_0)
            gaussian_ph_0 = self.gaussian_cal(x[8], self.avg_ph_0, self.std_ph_0)
            gaussian_sulfate_0 = self.gaussian_cal(x[9], self.avg_sulfate_0, self.std_sulfate_0)
            gaussian_alcohol_0 = self.gaussian_cal(x[10], self.avg_alcohol_0, self.std_alcohol_0)

            gaussian_fixed_acid_1 = self.gaussian_cal(x[0], self.avg_fixed_acid_1, self.std_fixed_acid_1)
            gaussian_vol_acid_1 = self.gaussian_cal(x[1], self.avg_vol_acid_1, self.std_vol_acid_1)
            gaussian_citric_acid_1 = self.gaussian_cal(x[2], self.avg_citric_acid_1, self.std_citric_acid_1)
            gaussian_resid_sugar_1 = self.gaussian_cal(x[3], self.avg_resid_sugar_1, self.std_resid_sugar_1)
            gaussian_chlorid_1 = self.gaussian_cal(x[4], self.avg_chlorid_1, self.std_chlorid_1)
            gaussian_free_sulf_dioxid_1 = self.gaussian_cal(x[5], self.avg_free_sulf_dioxid_1,
                                                            self.std_free_sulf_dioxid_1)
            gaussian_total_sulf_dioxid_1 = self.gaussian_cal(x[6], self.avg_total_sulf_dioxid_1,
                                                             self.std_total_sulf_dioxid_1)
            gaussian_density_1 = self.gaussian_cal(x[7], self.avg_density_1, self.std_density_1)
            gaussian_ph_1 = self.gaussian_cal(x[8], self.avg_ph_1, self.std_ph_1)
            gaussian_sulfate_1 = self.gaussian_cal(x[9], self.avg_sulfate_1, self.std_sulfate_1)
            gaussian_alcohol_1 = self.gaussian_cal(x[10], self.avg_alcohol_1, self.std_alcohol_1)

            proba_0 = gaussian_fixed_acid_0 * gaussian_vol_acid_0 * gaussian_citric_acid_0 * \
                      gaussian_resid_sugar_0 * gaussian_chlorid_0 * gaussian_free_sulf_dioxid_0 * \
                      gaussian_total_sulf_dioxid_0 * gaussian_density_0 * gaussian_ph_0 * \
                      gaussian_sulfate_0 * gaussian_alcohol_0

            proba_1 = gaussian_fixed_acid_1 * gaussian_vol_acid_1 * gaussian_citric_acid_1 * \
                      gaussian_resid_sugar_1 * gaussian_chlorid_1 * gaussian_free_sulf_dioxid_1 * \
                      gaussian_total_sulf_dioxid_1 * gaussian_density_1 * gaussian_ph_1 * \
                      gaussian_sulfate_1 * gaussian_alcohol_1

            if proba_0 >= proba_1:
                return_value = '0'
            else:
                return_value = '1'

        elif self.dataset_key == 'abalone':
            p_label_0 = self.proba_catego_var(x[0], 0)
            gaussian_length_0 = self.gaussian_cal(x[1], self.avg_length_0, self.std_length_0)
            gaussian_diameter_0 = self.gaussian_cal(x[2], self.avg_diameter_0, self.std_diameter_0)
            gaussian_height_0 = self.gaussian_cal(x[3], self.avg_height_0, self.std_height_0)
            gaussian_whol_weight_0 = self.gaussian_cal(x[4], self.avg_whol_weight_0, self.std_whol_weight_0)
            gaussian_shuc_weight_0 = self.gaussian_cal(x[5], self.avg_shuc_weight_0, self.std_shuc_weight_0)
            gaussian_visce_weight_0 = self.gaussian_cal(x[6], self.avg_visce_weight_0, self.std_visce_weight_0)
            gaussian_shell_weight_0 = self.gaussian_cal(x[7], self.avg_shell_weight_0, self.std_shell_weight_0)

            p_label_1 = self.proba_catego_var(x[0], 1)
            gaussian_length_1 = self.gaussian_cal(x[1], self.avg_length_1, self.std_length_1)
            gaussian_diameter_1 = self.gaussian_cal(x[2], self.avg_diameter_1, self.std_diameter_1)
            gaussian_height_1 = self.gaussian_cal(x[3], self.avg_height_1, self.std_height_1)
            gaussian_whol_weight_1 = self.gaussian_cal(x[4], self.avg_whol_weight_1, self.std_whol_weight_1)
            gaussian_shuc_weight_1 = self.gaussian_cal(x[5], self.avg_shuc_weight_1, self.std_shuc_weight_1)
            gaussian_visce_weight_1 = self.gaussian_cal(x[6], self.avg_visce_weight_1, self.std_visce_weight_1)
            gaussian_shell_weight_1 = self.gaussian_cal(x[7], self.avg_shell_weight_1, self.std_shell_weight_1)

            p_label_2 = self.proba_catego_var(x[0], 2)
            gaussian_length_2 = self.gaussian_cal(x[1], self.avg_length_2, self.std_length_2)
            gaussian_diameter_2 = self.gaussian_cal(x[2], self.avg_diameter_2, self.std_diameter_2)
            gaussian_height_2 = self.gaussian_cal(x[3], self.avg_height_2, self.std_height_2)
            gaussian_whol_weight_2 = self.gaussian_cal(x[4], self.avg_whol_weight_2, self.std_whol_weight_2)
            gaussian_shuc_weight_2 = self.gaussian_cal(x[5], self.avg_shuc_weight_2, self.std_shuc_weight_2)
            gaussian_visce_weight_2 = self.gaussian_cal(x[6], self.avg_visce_weight_2, self.std_visce_weight_2)
            gaussian_shell_weight_2 = self.gaussian_cal(x[7], self.avg_shell_weight_2, self.std_shell_weight_2)

            proba_0 = p_label_0 * gaussian_length_0 * gaussian_diameter_0 * gaussian_height_0 * gaussian_whol_weight_0 * gaussian_shuc_weight_0 * gaussian_visce_weight_0 * gaussian_shell_weight_0
            proba_1 = p_label_1 * gaussian_length_1 * gaussian_diameter_1 * gaussian_height_1 * gaussian_whol_weight_1 * gaussian_shuc_weight_1 * gaussian_visce_weight_1 * gaussian_shell_weight_1
            proba_2 = p_label_2 * gaussian_length_2 * gaussian_diameter_2 * gaussian_height_2 * gaussian_whol_weight_2 * gaussian_shuc_weight_2 * gaussian_visce_weight_2 * gaussian_shell_weight_2

            if (proba_0 >= proba_1) and (proba_0 >= proba_2):
                return_value = '0.0'
            elif (proba_1 >= proba_0) and (proba_1 >= proba_2):
                return_value = '1.0'
            else:
                return_value = '2.0'

        return return_value

    def proba_catego_var(self, letter, index):

        prob_categ_var = self.categ_vars_probas[letter][index]

        return prob_categ_var

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
                    else:
                        count_false_1 = count_false_1 + 1
                else:
                    if label_predicted == '0':
                        count_true_0 = count_true_0 + 1
                    else:
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
            confusion_matrix, accuracy, classification_report = helper_functions.compute_report_iris(y,
                                                                                                     labels_predicted)
        elif self.dataset_key == 'wine':
            confusion_matrix, accuracy, classification_report = helper_functions.compute_report_wine(y,
                                                                                                     labels_predicted)
        elif self.dataset_key == 'abalone':
            confusion_matrix, accuracy, classification_report = helper_functions.compute_report_abalone(y,
                                                                                                     labels_predicted)
        print("\nAccuracy: ", accuracy)
        print("\n")
        print("\nClassification report:\n", classification_report)
        print("\nConfusion matrix:\n", confusion_matrix)

        print("\nTotal count train data : ", len(self.train_labels))
        print("\nTotal count test data : ", len(y))

    def sum_cal_abalone(self, test_label):
        count_label_0 = 0
        count_label_1 = 0
        count_label_2 = 0

        for label in test_label:
            if label == '0.0':
                count_label_0 = count_label_0 + 1

            elif label == '1.0':
                count_label_1 = count_label_1 + 1

            elif label == '2.0':
                count_label_2 = count_label_2 + 1

        return count_label_0, count_label_1, count_label_2
