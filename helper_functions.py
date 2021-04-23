import numpy as np

def compute_report_iris(y_expected, y_predicted):
    unique = list(set(y_expected))
    unique.sort()
    length_unique = len(unique)
    matrix = [list() for x in range(len(unique))]

    for i in range(length_unique):
        matrix[i] = [0 for x in range(length_unique)]

    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i

    for i in range(len(y_expected)):
        x = lookup[y_expected[i]]
        y = lookup[y_predicted[i]]
        matrix[x][y] += 1

    confusion_matrix = np.asarray(matrix)

    index_setosa = lookup['Iris-setosa']
    index_versicolor = lookup['Iris-versicolor']
    index_virginica = lookup['Iris-virginica']

    tp_setosa = matrix[index_setosa][index_setosa]
    tp_versicolor = matrix[index_versicolor][index_versicolor]
    tp_virginica = matrix[index_virginica][index_virginica]

    tn_setosa = matrix[index_virginica][index_virginica] + matrix[index_virginica][index_versicolor] + matrix[index_versicolor][index_virginica] + matrix[index_versicolor][index_versicolor]
    tn_versicolor = matrix[index_setosa][index_setosa] + matrix[index_setosa][index_virginica] + matrix[index_virginica][index_setosa] + matrix[index_virginica][index_virginica]
    tn_virginica = matrix[index_setosa][index_setosa] + matrix[index_setosa][index_versicolor] + matrix[index_versicolor][index_setosa] + matrix[index_versicolor][index_versicolor]

    fp_setosa = matrix[index_versicolor][index_setosa] + matrix[index_virginica][index_setosa]
    fp_versicolor = matrix[index_setosa][index_versicolor] + matrix[index_virginica][index_versicolor]
    fp_virginica = matrix[index_setosa][index_virginica] + matrix[index_versicolor][index_virginica]

    fn_setosa = matrix[index_setosa][index_versicolor] + matrix[index_setosa][index_virginica]
    fn_versicolor = matrix[index_versicolor][index_setosa] + matrix[index_versicolor][index_virginica]
    fn_virginica = matrix[index_virginica][index_setosa] + matrix[index_virginica][index_versicolor]

    accuracy = (tp_setosa + tp_versicolor + tp_virginica) / len(y_expected)

    precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
    precision_versicolor = tp_versicolor/ (tp_versicolor + fp_versicolor)
    precision_virginica = tp_virginica/ (tp_virginica + fp_virginica)

    recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
    recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
    recall_virginica = tp_virginica / (tp_virginica + fn_virginica)

    f_score_setosa = 2*(recall_setosa * precision_setosa) / (recall_setosa + precision_setosa)
    f_score_versicolor = 2*(recall_versicolor * precision_versicolor) / (recall_versicolor + precision_versicolor)
    f_score_virginica = 2*(recall_virginica * precision_virginica) / (recall_virginica + precision_virginica)

    support_setosa, support_versicolor, support_virginica = sum_cal_iris(y_expected)

    classification_report = '-----SETOSA : \nPrecision: {0:.2f}, Recall: {1:.2f}, F1-score: {2:.2f}, Support: {3}. ' \
                   '\n-----VERSICOLOR : \nPrecision: {4:.2f}, Recall: {5:.2f}, F1-score: {6:.2f}, Support: {7}. ' \
                   '\n-----VIRGINICA : \nPrecision: {8:.2f}, Recall: {9:.2f}, F1-score: {10:.2f}, Support: {11}.' \
        .format(precision_setosa, recall_setosa, f_score_setosa, support_setosa, precision_versicolor, recall_versicolor,
                f_score_versicolor, support_versicolor, precision_virginica, recall_virginica, f_score_virginica, support_virginica)

    return confusion_matrix, accuracy, classification_report

def compute_report_wine(y_expected, y_predicted):
    unique = list(set(y_expected))
    unique.sort()
    length_unique = len(unique)
    matrix = [list() for x in range(len(unique))]

    for i in range(length_unique):
        matrix[i] = [0 for x in range(length_unique)]

    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i

    for i in range(len(y_expected)):
        x = lookup[y_expected[i]]
        y = lookup[y_predicted[i]]
        matrix[x][y] += 1

    confusion_matrix = np.asarray(matrix)

    index_0 = lookup['0']
    index_1 = lookup['1']

    tp_0 = matrix[index_0][index_0]
    fp_0 = matrix[index_0][index_1]

    tn_0 = matrix[index_1][index_1]
    fn_0 = matrix[index_1][index_0]

    tp_1 = matrix[index_1][index_1]
    fp_1 = matrix[index_1][index_0]

    tn_1 = matrix[index_0][index_0]
    fn_1 = matrix[index_0][index_1]

    accuracy = (tp_0 + tp_1) / len(y_expected)

    precision_0 = tp_0 / (tp_0 + fp_0)
    precision_1 = tp_1 / (tp_1 + fp_1)

    recall_0 = tp_0 / (tp_0 + fn_0)
    recall_1 = tp_1 / (tp_1 + fn_1)

    f_score_0 = 2*(recall_0 * precision_0) / (recall_0 + precision_0)
    f_score_1 = 2*(recall_1 * precision_1) / (recall_1 + precision_1)

    support_0, support_1 = sum_cal_wine(y_expected)

    classification_report = '-----0 : \nPrecision: {0:.2f}, Recall: {1:.2f}, F-score: {2:.2f}, Support: {3}. ' \
                   '\n-----1 : \nPrecision: {4:.2f}, Recall: {5:.2f}, F-score: {6:.2f}, Support: {7}. ' \
        .format(precision_0, recall_0, f_score_0, support_0, precision_1, recall_1,
                f_score_1, support_1)


    return confusion_matrix, accuracy, classification_report

def compute_report_abalone(y_expected, y_predicted):
    unique = list(set(y_expected))
    unique.sort()
    length_unique = len(unique)
    matrix = [list() for x in range(len(unique))]

    for i in range(length_unique):
        matrix[i] = [0 for x in range(length_unique)]

    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i

    for i in range(len(y_expected)):
        x = lookup[y_expected[i]]
        y = lookup[y_predicted[i]]
        matrix[x][y] += 1

    confusion_matrix = np.asarray(matrix)

    index_0 = lookup['0.0']
    index_1 = lookup['1.0']
    index_2 = lookup['2.0']

    tp_0 = matrix[index_0][index_0]
    tp_1 = matrix[index_1][index_1]
    tp_2 = matrix[index_2][index_2]

    tn_0 = matrix[index_2][index_2] + matrix[index_2][index_1] + matrix[index_1][index_2] + matrix[index_1][index_1]
    tn_1 = matrix[index_0][index_0] + matrix[index_0][index_2] + matrix[index_2][index_0] + matrix[index_2][index_2]
    tn_2 = matrix[index_0][index_0] + matrix[index_0][index_1] + matrix[index_1][index_0] + matrix[index_1][index_1]

    fp_0 = matrix[index_1][index_0] + matrix[index_2][index_0]
    fp_1 = matrix[index_0][index_1] + matrix[index_2][index_1]
    fp_2 = matrix[index_0][index_2] + matrix[index_1][index_2]

    fn_0 = matrix[index_0][index_1] + matrix[index_0][index_2]
    fn_1 = matrix[index_1][index_0] + matrix[index_1][index_2]
    fn_2 = matrix[index_2][index_0] + matrix[index_2][index_1]

    accuracy = (tp_0 + tp_1 + tp_2) / len(y_expected)

    precision_0 = tp_0 / (tp_0 + fp_0)
    precision_1 = tp_1/ (tp_1 + fp_1)
    precision_2 = tp_2/ (tp_2 + fp_2)

    recall_0 = tp_0 / (tp_0 + fn_0)
    recall_1 = tp_1 / (tp_1 + fn_1)
    recall_2 = tp_2 / (tp_2 + fn_2)

    f_score_0 = 2*(recall_0 * precision_0) / (recall_0 + precision_0)
    f_score_1 = 2*(recall_1 * precision_1) / (recall_1 + precision_1)
    f_score_2 = 2*(recall_2 * precision_2) / (recall_2 + precision_2)

    support_0, support_1, support_2 = sum_cal_abalone(y_expected)

    classification_report = '-----0 : \nPrecision: {0:.2f}, Recall: {1:.2f}, F-score: {2:.2f}, Support: {3}. ' \
                   '\n-----1 : \nPrecision: {4:.2f}, Recall: {5:.2f}, F-score: {6:.2f}, Support: {7}. ' \
                   '\n-----2 : \nPrecision: {8:.2f}, Recall: {9:.2f}, F-score: {10:.2f}, Support: {11}.' \
        .format(precision_0, recall_0, f_score_0, support_0, precision_1, recall_1,
                f_score_1, support_1, precision_2, recall_2, f_score_2, support_2)


    return confusion_matrix, accuracy, classification_report

def sum_cal_iris(test_label):
    count_label_setosa = 0
    count_label_versicolor = 0
    count_label_virginica = 0

    for label in test_label:
        if label == 'Iris-setosa':
            count_label_setosa = count_label_setosa + 1

        elif label == 'Iris-versicolor':
            count_label_versicolor = count_label_versicolor + 1

        elif label == 'Iris-virginica':
            count_label_virginica = count_label_virginica + 1

    return count_label_setosa, count_label_versicolor, count_label_virginica

def sum_cal_wine(test_label):
    count_label_0 = 0
    count_label_1 = 0

    for label in test_label:
        if label == '0':
            count_label_0 = count_label_0 + 1
        else:
            count_label_1 = count_label_1 + 1

    return count_label_0, count_label_1

def sum_cal_abalone(test_label):
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

def get_unique_item(vector):
    list_items = [i[0] for i in vector]
    unique = set(list_items)
    lookup = dict()

    for i, value in enumerate(unique):
        lookup[value] = i

    return lookup