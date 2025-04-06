import argparse
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LassoCV
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, auc, recall_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
import itertools
import os
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler
import pylab as pl
from sklearn.multiclass import OneVsRestClassifier
from yellowbrick.classifier import ROCAUC
from dtreeviz import dtreeviz
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfeature_matrix", required=True)
    parser.add_argument("--candidategenes", required=True)
    parser.add_argument("--n_estimators", required=True, type=int)
    parser.add_argument("--max_depth", required=True, type=int)
    parser.add_argument("--min_samples_split", required=True, type=int)
    parser.add_argument("--min_samples_leaf", required=True, type=int)
    parser.add_argument("--max_features", required=True, type=str, default=None)
    parser.add_argument("--bootstrap", required=True, type=str)
    args = parser.parse_args()

    if args.max_features == 'None':
        args.max_features = None

    args.bootstrap = args.bootstrap.lower() in ['true', '1', 'yes']

    feat = args.inputfeature_matrix
    candidate_genes = args.candidategenes
    candidate_genes_df = pd.read_csv(candidate_genes, sep='\t', usecols=['Gene'])
    df = pd.read_csv(feat, sep='\t')

    df = df[df['genes'].notna()]
    df = df.set_index(['genes'])

    X_test = df[df['Disease_Association'].isna()].drop(['Disease_Association'], axis=1)

    df = df[df['Disease_Association'].notna()]

    factor = pd.factorize(df['Disease_Association'], sort=True)
    df.Disease_Association = factor[0]
    definitions = factor[1]

    Y = df[['Disease_Association']]
    X = df.drop(['Disease_Association'], axis=1)

    X = X.fillna(0)
    X_test = X_test.fillna(0)

    imp = IterativeImputer(max_iter=10, random_state=0)
    X_values = imp.fit_transform(X.values)
    X_test_values = imp.transform(X_test.values)

    scaler = MinMaxScaler()
    X_values = scaler.fit_transform(X_values)
    X_test_values = scaler.transform(X_test_values)

    X = pd.DataFrame(X_values, index=X.index, columns=X.columns)
    X_test = pd.DataFrame(X_test_values, index=X_test.index, columns=X_test.columns)
    
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    counter = 0
    validation = []
    validation_prob_0 = []
    validation_prob_1 = []

    true_y = pd.DataFrame()

    genes = []
    labels = []

    for train_index, test_index in kf.split(X, Y):
        counter += 1
        X_train, X_validate = X.iloc[train_index], X.iloc[test_index]
        y_train, y_validate = Y.iloc[train_index], Y.iloc[test_index]
        print(y_train.Disease_Association.value_counts())

        rf_predictions, rf_probs_0, rf_probs_1 = cross_validate(X_train, X_validate, y_train, args)

        validation.extend(rf_predictions)
        validation_prob_0.extend(rf_probs_0)
        validation_prob_1.extend(rf_probs_1)

        genes.extend(X_validate.index)
        labels.extend(y_validate["Disease_Association"].tolist())

        true_y = true_y.append(y_validate)

    true_y_list = true_y["Disease_Association"].tolist()
    validation_prob_1_list = validation_prob_1

    with open("prc_data.txt", "w") as f:
        f.write("True_Label\tPredicted_Prob\n")
        for true_label, prob in zip(true_y_list, validation_prob_1_list):
            f.write(f"{true_label}\t{prob}\n")

    test_rf_predictions, test_rf_probs_0, test_rf_probs_1 = predict(X, Y, X_test, args)

    df_test_pred = pd.DataFrame({'Gene': list(X_test.index), 'predictions': test_rf_predictions, str(definitions[0]): test_rf_probs_0, str(definitions[1]): test_rf_probs_1})
    df_test_pred.to_csv('predictions.tsv', sep='\t', index=False)

    fpr, tpr, threshold = roc_curve(true_y["Disease_Association"].tolist(), validation_prob_1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('roc_curve.png', bbox_inches='tight')

    precision, recall, prc_thresholds = precision_recall_curve(true_y["Disease_Association"].tolist(), validation_prob_1)

    average_precision = average_precision_score(true_y["Disease_Association"].tolist(), validation_prob_1)

    prc_auc = auc(recall, precision)

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (average precision = %0.2f, AUC = %0.2f)' % (average_precision, prc_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig('prc_curve.png', bbox_inches='tight')



def cross_validate(X_train, X_validate, y_train, args):

    rf_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        random_state=42
    )
    rf_model.fit(X_train, y_train.values.ravel())

    rf_predictions = rf_model.predict(X_validate)
    rf_probs = rf_model.predict_proba(X_validate)
    rf_probs_0 = rf_probs[:, 0]
    rf_probs_1 = rf_probs[:, 1]

    return rf_predictions, rf_probs_0, rf_probs_1

def predict(X, Y, X_test, args):

    rf_model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        random_state=42
    )
    rf_model.fit(X, Y.values.ravel())
    importances = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances,
        'Standard_Deviation': std
    })

    feature_importance_df.to_csv('forest_importances_with_std.tsv', sep='\t', index=False)

    rf_predictions = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    rf_probs_0 = rf_probs[:, 0]
    rf_probs_1 = rf_probs[:, 1]

    return rf_predictions, rf_probs_0, rf_probs_1

if __name__ == "__main__":
    main()
