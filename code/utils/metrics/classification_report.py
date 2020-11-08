import numpy as np
from pprint import pformat
from sklearn.metrics import classification_report

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:41:09 2020

@author: dylan_monfret
"""

" Local Function "


def di_calculation(genres_dict):
    """
    :param genres_dict: Dictionary of genre accounting by label
    :return: the Disparate Impact mesure, a measure of the imbalance between 2 sub-group in a given group (here 'M' and
    'F' for a specific job label).
    """

    if not genres_dict:
        return 0

    try:  # Disparate Impact definition.
        m_count = genres_dict["M"]
        f_count = genres_dict["F"]
        the_di = max(m_count, f_count) / min(m_count, f_count)
        return the_di

    except KeyError:  # Could happen essentially if a group is "empty".

        if "M" not in genres_dict.keys():
            return genres_dict["F"]

        elif "F" not in genres_dict.keys():
            return genres_dict["M"]


" Class definition "


class Report:
    def __init__(self, array_result):
        """
        'Report' is a classification_report dictionary type from 'sklearn.metrics', but which include the Disparate
        Impact for each class / label.

        :param array_result: numpy.array from a machine learning process with 3 distinct elements: the first containing
        real test values, the second containing predicted values and the last representing the sensitive genre variable
        of modality 'F' or 'M'. For a correct method's execution, array's elements must be kept in order.
        """

        # macros DI init
        true_di_sum = float()
        pred_di_sum = float()
        diff_di_sum = float()

        # Extraction
        true = array_result[0]
        pred = array_result[1]
        genres = array_result[2]
        label_list = list(set(true))
        the_len = len(label_list)

        # Basic classification_report to enhance with DIs
        self.report = classification_report(true, pred, labels=label_list, output_dict=True, zero_division='warn')

        for label in label_list:
            # Count 'M' / 'F' apparition for a specific label
            # Label filtering made with numpy.where
            unique1, counts1 = np.unique(genres[np.where(true == label)], return_counts=True)
            unique2, counts2 = np.unique(genres[np.where(pred == label)], return_counts=True)
            true_dict_di = dict(zip(unique1, counts1))
            pred_dict_di = dict(zip(unique2, counts2))

            # 'di_calculation' call and DI difference calculation
            true_di = di_calculation(true_dict_di)
            pred_di = di_calculation(pred_dict_di)
            diff_di = abs(true_di - pred_di)

            # Results added to former classification_report dictionary in specific label section
            self.report[label]["true_di"] = true_di
            self.report[label]["pred_di"] = pred_di
            self.report[label]["diff_di"] = diff_di

            # DIs summed for DIs macros calculation
            true_di_sum += true_di
            pred_di_sum += pred_di
            diff_di_sum += diff_di

        # DIs macros added to former classification_report dictionary in 'macro avg' section
        self.report["macro avg"]["true_di"] = true_di_sum / the_len
        self.report["macro avg"]["pred_di"] = pred_di_sum / the_len
        self.report["macro avg"]["diff_di"] = diff_di_sum / the_len

    def __repr__(self):
        """
        :return: The report's dictionary (self.report), but readable thanks to pprint.pformat.
        """
        return pformat(self.report)


" Test "

if __name__ == "__main__":
    import random as rdm

    rdm.seed(42069)

    genre_binary = {0: "M", 1: "F"}

    n = 1e3
    n_job = 29

    test_true = [rdm.randint(1, n_job) for i in range(int(n))]
    test_pred = test_true[0:int(n / 2)] + [rdm.randint(1, n_job) for j in range(int(n / 2))]
    test_genr_binary = [rdm.randint(0, 1) for k in range(int(n))]
    test_genr = [genre_binary[i] for i in test_genr_binary]

    arr = np.array([test_true, test_pred, test_genr])

    a_report = Report(arr)
    print(a_report)
    print("")
    print(a_report.report)
    print("")
    print(a_report.report["macro avg"])
    print("")
    try:
        print(a_report.report["1"])
    except KeyError:
        print("Le label '1' n'existe pas.")
