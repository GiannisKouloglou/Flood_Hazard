# Methods to Fit and Evaluate a Machine Learning models including preprocessing actions
#
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay

from sklearn.preprocessing import MultiLabelBinarizer, Binarizer
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import random


# -------------------------------------------------------------------------------------------
# Preprocess dataset:
#   a) Sampling over the majority category
#   b) Scale the dataset using MinMax Scaller
#   c) Split dataset to training and testing based on perc (decimal number)
#
def preprocess_old( df, attrbs, targs, categories_cardinality, perc ):

    print("\n =========== Start PREPROCESSING process ============== \n")

    # Take a sample of dataset df based on the following steps:
    # a) Find the category with maximum cardinality
    max_card = categories_cardinality['No Hazard'][0]
    name_categ = 'No Hazard'
    sum = 0
    for item in categories_cardinality:
        sum = sum + categories_cardinality[item][0]
        if categories_cardinality[item][0] >= max_card:
            max_card = categories_cardinality[item][0]
            name_categ = item

    perc_max_categ = round((max_card*100)/sum, 2)
    print(" Max cardinality = ", max_card, " Item = ", name_categ, " Total = ", sum, " and perc_max_categ = ", perc_max_categ)

    # b) Choose a sample of data from category with maximum cardinality
    ratio = round( 1.0 - max(perc_max_categ/100, perc/100), 3 )
    print(" Choose a sample of the max category equal with: ", ratio)

    # Create new dataset based on the sampling
    sampling_df = pd.DataFrame( df.loc[ df['Hazard_Category'] == name_categ ].sample(frac=ratio) )
    print( " Sampling shape = ", sampling_df.shape )

    # Add the remaining rows of other categories to sampling_df
    remain_categ = []
    for k in categories_cardinality:
       if k != name_categ:
           remain_categ.append(k)
           temp = pd.DataFrame( df.loc[ df['Hazard_Category'] == k ] )
           print(" new temp: ", temp.shape )

           sampling_df = pd.concat( [ sampling_df, temp ], axis=0, ignore_index=False)

    # print(" Remain categories: ", remain_categ)
    print("\n---------------->>>>>>>>>>>>>>>>>>>>>>>> ")
    print(" sampling_df shape = ", sampling_df.shape )
    print(" Class distribution in sampling_df: ")
    print( " No hazard = ", sampling_df.loc[ sampling_df['Hazard_Category'] == 'No Hazard' ].shape  )
    print( " Low hazard = ", sampling_df.loc[ sampling_df['Hazard_Category'] == 'Low Hazard' ].shape  )
    print( " Medium hazard = ", sampling_df.loc[ sampling_df['Hazard'] == 0.8 ].shape )
    print( " High hazard = ", sampling_df.loc[ sampling_df['Hazard'] == 1.0].shape)
    print("<<<<<<<<<<<<<<<<<<<<<<<<------------------------------\n ")

    # Scale the dataset for Machine Learning analysis
    min_max_scaler = preprocessing.MinMaxScaler()
    scale_sampling_df = pd.DataFrame( min_max_scaler.fit_transform( sampling_df.loc[:, attrbs ] ), columns=attrbs, index=sampling_df.index )

    print( " scale_sampling_df.shape = ", scale_sampling_df.shape )

    # fname_scale_sampl = 'ScaleSampling.csv'
    # scale_sampling_df.to_csv(fname_scale_sampl, index = True)
    # print( scale_sampling_df.head )

    # Training set
    X = pd.DataFrame( scale_sampling_df.loc[:, attrbs ] )

    # Testing set
    y = pd.DataFrame( sampling_df.loc[:, targs ] )


    # Due to the dataset does not have a balanced number of examples for each class label, it splits into train and test sets
    # in a way that preserves the same proportions of examples in each class as observed in the original dataset (stratify=y)
    # For comparison machine learning algorithms, it is desirable that they are fit and evaluated on the same subsets of the dataset.
    # Thus random_state=1, otherwise for randomly split random_state=None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = perc, random_state=1, stratify=y)  # [X, y]

    return X_train, X_test, y_train, y_test, min_max_scaler

def preprocess( df, attrbs, targs, perc ):

    print("\n =========== Start PREPROCESSING process ============== \n")

    # Scale the dataset for Machine Learning analysis
    min_max_scaler = preprocessing.MinMaxScaler()
    scale_sampling_df = pd.DataFrame( min_max_scaler.fit_transform( df.loc[:, attrbs ] ), columns=attrbs, index=df.index )

    print( " scale_sampling_df.shape = ", scale_sampling_df.shape )
    # Training set
    X = pd.DataFrame( scale_sampling_df.loc[:, attrbs ] )

    # Testing set
    y = pd.DataFrame( df.loc[:, targs ] )


    # Due to the dataset does not have a balanced number of examples for each class label, it splits into train and test sets
    # in a way that preserves the same proportions of examples in each class as observed in the original dataset (stratify=y)
    # For comparison machine learning algorithms, it is desirable that they are fit and evaluated on the same subsets of the dataset.
    # Thus random_state=1, otherwise for randomly split random_state=None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = perc, random_state=1, stratify=y)  # [X, y]

    return X_train, X_test, y_train, y_test, min_max_scaler
# ---------------------------------------------------------------------------------------------------------------------------------
# Create a ML model with Grid Search for parameter fine tuning and deliver the best one in terms of the evaluation criteria.
#
def create_ML_models(params, cv_n_folds, pipes, X_train, X_test, y_train, y_test, targ_categ, outpt_fname, model_name, dict_param, scaler):
    # Opening file to write experiment results
    file_experiment_results = open(outpt_fname, 'a+')

    print("\n ****************** inside create_ML_models...")
    print("\n length parameters:", len(params))
    print(params)
    print(pipes)

    gs = GridSearchCV(pipes, params, cv=cv_n_folds)  # ,  refit=True)
    gs = gs.fit(X_train, y_train.values.ravel())
    predicted = gs.predict(X_test).tolist()

    # Take the best model according to the grid search process and estimate its predictions over the X_test
    best_model = gs.best_estimator_
    best_model.fit(X_train, y_train.values.ravel())
    y_pred_best = best_model.predict(X_test)

    print("\n", file=file_experiment_results)
    print("Grid scores on development set:", file=file_experiment_results)
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']

    for params, mean, std in zip(gs.cv_results_['params'], means, stds):
        print("For %r, the accuracy is: %0.3f (+/-%0.03f) " % (params, mean, std * 2), file=file_experiment_results)

    print("\n The model is trained on the full development set.", file=file_experiment_results)
    print("The scores are computed on the full evaluation set. \n", file=file_experiment_results)
    print(" -------------------------------------------------- \n", file=file_experiment_results)

    print("Detailed classification report over the best set of parameters and best model:",
          file=file_experiment_results)
    print("\n The best model has parameters and score: ", file=file_experiment_results)
    print(gs.best_params_, '\t', gs.best_score_, '\n', file=file_experiment_results)

    measures = classification_report(y_test, y_pred_best, target_names=targ_categ, digits=2, output_dict=False)
    print(measures, file=file_experiment_results)

    file_experiment_results.close()

    ML_results = {'Model': best_model, 'Targets_predicted_best': y_pred_best, 'Evaluation_Measures': measures, 'Grid_Search_Results': gs, 'Scaler':scaler}
    print('[INFO] Saving model...')
    pickle.dump(ML_results, open(model_name, 'wb'))
    print('[INFO] Model is saved with name:', model_name)

    dict_param['model'] = model_name
    dict_param['score'] = gs.best_score_

    file_experiment_results.close()

    return ML_results, dict_param

# ---------------------------------------------------------------------------------------------------------------------------------
# Evaluate the ML model creating a Confusion Matrix
def evaluate_confusion_matrix(y_test, y_pred, targ_categ, outpt_fname, model_name, path ):

    # Opening file to write experiment results
    # file_experiment_results = open(outpt_fname, 'a+')

    print("\n ===>>> CHECKING REASONS inside evaluate_confusion_matrix ======")
    # print("\n file_experiment_results=", file_experiment_results)

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_DF = pd.DataFrame( conf_mat, columns=targ_categ, index=targ_categ )

    # Print results to file
    # print("\n ====== CONFUSION MATRIX ======", file=file_experiment_results)
    # print(conf_mat, file=file_experiment_results)

    # Plot non-normalized confusion matrix
    cm_display = ConfusionMatrixDisplay(conf_mat, display_labels=targ_categ)
    cm_display = cm_display.plot(include_values=True, cmap='Blues', ax=None, values_format='d', xticks_rotation='horizontal', colorbar=False)

    # plt.show()

    fig_name = path + 'Confusion_Matrix' + '_' + model_name + '.png'
    print(" \n Location of the Confusion Matrix is: ", fig_name)
    plt.savefig(fig_name)

    plt.close()

    # file_experiment_results.close()

    print(conf_mat_DF)
    print("\nEXIT evaluate_confusion_matrix\n")

    return conf_mat, conf_mat_DF


#
# Obsolete the display function does not work properly
#
# def evaluate_confusion_matrix(y_test, y_pred, targ_categ, outpt_fname, model_name, path ):
#
#     # Opening file to write experiment results
#     file_experiment_results = open(outpt_fname, 'a+')
#
#     print("\n ===>>> CHECKING REASONS inside evaluate_confusion_matrix ======")
#     print("\n file_experiment_results=", file_experiment_results)
#
#     # Confusion matrix
#     conf_mat = confusion_matrix(y_test, y_pred)
#
#     print(conf_mat)
#     print(" <<<<<<<<<================== \n")
#
#     print("\n ====== CONFUSION MATRIX ======", file=file_experiment_results)
#     print(conf_mat, file=file_experiment_results)
#
#     # Plot non-normalized confusion matrix
#     # plt.figure(figsize=(200,150))
#     cm_display = ConfusionMatrixDisplay(conf_mat, display_labels=targ_categ).plot(cmap="Blues")
#
#     print(cm_display)
#
#     # tick_marks = np.arange(len(targ_categ))
#     # plt.xticks(tick_marks, rotation=45,  fontsize=11)
#     # plt.yticks(tick_marks, targ_categ,  fontsize=11)
#     # plt.rcParams["font.size"] = "12"
#     #
#     fig_name = path + 'Confusion_Matrix' + '_' + model_name + '.png'
#     print(" \n Location of the Confusion Matrix is: ", fig_name)
#     plt.savefig(fig_name)
#
#     # Display Confusion Matrix
#     # plt.show()
#     # plt.show(cm_display.figure_)
#
#     # plt.close()
#     file_experiment_results.close()
#
#     return conf_mat

def cardinality(df):
    cardinality_dic = {
        'Low Hazard': df.loc[(df['Water_Velocity'] <= 1.0) & (df['Water_Depth'] < 1.0)].index.tolist(),
        'Medium Hazard': df.loc[(df['Water_Velocity'] <= 1.0) & (df['Water_Depth'] >= 1.0)].index.tolist(),
        'High Hazard': df.loc[df['Water_Velocity'] > 1.0].index.tolist()
    }
    # a) Find the category with maximum cardinality
    max_card = len(cardinality_dic['Low Hazard'])
    name_categ = 'Low Hazard'
    sum = 0
    for item in cardinality_dic:
        sum = sum + len(cardinality_dic[item])
        if len(cardinality_dic[item]) >= max_card:
            max_card = len(cardinality_dic[item])
            name_categ = item

    perc_max_categ = round((max_card * 100) / sum, 2)
    print(" Max cardinality = ", max_card, " Item = ", name_categ, " Total = ", sum, " and perc_max_categ = ",
          perc_max_categ)

    # b) Choose a sample of data from category with maximum cardinality
    ratio = round(1.0 - max(perc_max_categ / 100, 0.3 / 100), 5)
    print(" Choose a sample of the max category equal with: ", ratio)

    # Create new dataset based on the sampling
    sampling_list = random.sample(cardinality_dic[name_categ], int(round(ratio * sum, 0)))
    sampling_df = pd.DataFrame(df.loc[sampling_list])
    print(" Sampling shape Hazard= ", sampling_df.shape)

    # Add the remaining rows of other categories to sampling_df
    remain_categ = []
    for k in cardinality_dic:
        if k != name_categ:
            remain_categ.append(k)
            temp = pd.DataFrame(df.loc[cardinality_dic[k]])
            print(" new temp: ", temp.shape)

            sampling_df = pd.concat([sampling_df, temp], axis=0, ignore_index=False)

    return sampling_df