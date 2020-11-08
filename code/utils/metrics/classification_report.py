import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:41:09 2020

@author: dylan_monfret
"""

# pd.set_option("display.max_rows", None, "display.max_columns", None)

" Local Function "


def di_calculation(genres_dict):
    """
    Function to calculate Disparate Impact.

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


def report_dict_creator(sklearn_report, true_values, pred_values, genre_values, a_label_list):
    """
    A function to create enhanced report including DI and macro averages.

    :param sklearn_report: basic sklearn.classification_report dictionary.
    :param true_values: y_true
    :param pred_values: y_pred
    :param genre_values: vector of genres
    :param a_label_list: list of possible value taken by y_true / y_pred
    :return: the enhanced dictionary report including DIs and other macro averages
    """
    # macros DI init
    true_di_sum = float()
    pred_di_sum = float()
    diff_di_sum = float()
    the_len = len(a_label_list)

    for label in a_label_list:
        # Count 'M' / 'F' apparition for a specific label
        # Label filtering made with numpy.where
        unique1, counts1 = np.unique(genre_values[np.where(true_values == label)], return_counts=True)
        unique2, counts2 = np.unique(genre_values[np.where(pred_values == label)], return_counts=True)
        true_dict_di = dict(zip(unique1, counts1))
        pred_dict_di = dict(zip(unique2, counts2))

        # 'di_calculation' call and DI difference calculation
        true_di = di_calculation(true_dict_di)
        pred_di = di_calculation(pred_dict_di)
        diff_di = abs(true_di - pred_di)

        # Results added to former classification_report dictionary in specific label section
        sklearn_report[label]["true_di"] = true_di
        sklearn_report[label]["pred_di"] = pred_di
        sklearn_report[label]["diff_di"] = diff_di

        # DIs summed for DIs macros calculation
        true_di_sum += true_di
        pred_di_sum += pred_di
        diff_di_sum += diff_di

    # DIs macros added to former classification_report dictionary in 'macro avg' section
    sklearn_report["macro avg"]["true_di"] = true_di_sum / the_len
    sklearn_report["macro avg"]["pred_di"] = pred_di_sum / the_len
    sklearn_report["macro avg"]["diff_di"] = diff_di_sum / the_len

    return sklearn_report


def report_df_creator(enhanced_report):
    """
    A function to transform classification report dictionary into a clean pandas.DataFrame.

    :param enhanced_report: classification report dictionary with DIs.
    :return: classification report df sorted by index.
    """

    # Change the dictionary into a pandas.DataFrame
    report_df = pd.DataFrame(enhanced_report).transpose()

    # Sort by index with labels first (need addition then deletion of leading zeros)
    # For our problem, only 1 leading zero by leading are necessary
    report_df.index = report_df.index.map(lambda x: '{0:0>2}'.format(x))
    report_df = report_df.sort_index()
    report_df.index = report_df.index.map(lambda x: '{}'.format(x.lstrip("0")))

    # Filling of accuracy line with 'blank' cell
    report_df.loc["accuracy"] = [np.NaN, np.NaN, enhanced_report["accuracy"], np.NaN, np.NaN, np.NaN, np.NaN]

    # Addition of some weighted averages (DIs and DIs differences)
    report_df.loc["weighted avg", "true_di"] = (
            report_df.true_di[:-3] * report_df.support[:-3] / report_df.support[:-3].sum()).sum()
    report_df.loc["weighted avg", "pred_di"] = (
            report_df.pred_di[:-3] * report_df.support[:-3] / report_df.support[:-3].sum()).sum()
    report_df.loc["weighted avg", "diff_di"] = (
            report_df.diff_di[:-3] * report_df.support[:-3] / report_df.support[:-3].sum()).sum()

    return report_df


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

        # Extraction
        true = array_result[0]
        pred = array_result[1]
        genres = array_result[2]
        label_list = list(set(true).union(set(pred)))

        # Basic classification_report to enhance with DIs
        basic_report = classification_report(true, pred, labels=label_list, output_dict=True, zero_division='warn')

        # Report dictionary created with DI calculated
        self.report_dict = report_dict_creator(basic_report, true, pred, genres, label_list)

        # Report dataframe created with previous dictionary
        self.report_df = report_df_creator(self.report_dict)

    def __repr__(self):
        """
        :return: The report's pandas.DataFrame (self.report_df) as a string.
        """
        return self.report_df.to_string()


" Test "

if __name__ == "__main__":
    import random as rdm
    import pprint

    rdm.seed(42069)

    genre_binary = {0: "M", 1: "F"}

    n = 2.5e4
    n_job = 29

    test_true = [rdm.randint(1, n_job) for i in range(int(n))]
    test_pred = test_true[0:int(n / 2)] + [rdm.randint(1, n_job) for j in range(int(n / 2))]
    test_genr_binary = [rdm.randint(0, 1) for k in range(int(n))]
    test_genr = [genre_binary[i] for i in test_genr_binary]

    arr = np.array([test_true, test_pred, test_genr])

    a_report = Report(arr)
    print(a_report)
    print("")
    pprint.pprint(a_report.report_dict)
    print("")
    print(a_report.report_df)
