import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .processing_data_MOF import *

"""
Lucía Moran and Hannes Kneiding's CustomDataset class.
"""


def find_accumulated_relevance(df, acc_rel_lb, start_relevance):
    """Finds the relevance bound given an accumulated relevance lower bound.
    Arguments
        df,                 pd.DataFrame
            DataFrame containing relevances of features.
        acc_rel_lb,         float
            Lower bound for accumulated relevance.
        start_relevance,    float
            Start point for finding the accumulated relevance.

    Returns
        relevance,          float
            Relevance value corresponding to the lower bound
            for the accumulated relevance.
    """
    if start_relevance > 1e-3:
        eps = 1e-4
    else:
        eps = 1e-6
    relevance = start_relevance - eps
    counter = 0
    while np.sum(df["relevance"][df["relevance"]>relevance])< acc_rel_lb:
        relevance -= eps
        counter += 1

    return relevance

def load_specific_split(path_to_ac, path_to_target, target, path_to_relevance, relevance=1e-5):
    """
    Loads the data split in split_data/
    """

    df_target = pd.read_csv(path_to_target)
    df = pd.read_csv(path_to_ac)
    target_vector = []
    for i, name1 in enumerate(df["id"]):
        for j, name2 in enumerate(df_target["id"]):
            if name1 == name2:
                target_vector.append(df_target[target][j])

    df = df.dropna(axis=1)
    df["target"] = target_vector
    """
    Loading specific split
    """
    path_to_split = "/home/jeb/Desktop/AABBA_Paper/data_preprocessing/data_split/"
    training_names = pd.read_csv(path_to_split + "file_name_train.csv")
    test_names = pd.read_csv(path_to_split + "file_name_test.csv")
    val_names = pd.read_csv(path_to_split + "file_name_val.csv")
    xtrain = df[df["id"].isin(training_names["names"])]
    xval = df[df["id"].isin(val_names["names"])]
    xtest = df[df["id"].isin(test_names["names"])]
    ytrain = xtrain["target"]; yval = xval["target"]; ytest = xtest["target"]
    if relevance < 0:
        """
        If the relevance is a negative number then we only drop the redundant features.
        """
        xtrain = xtrain.drop(columns=["target", "id", "Unnamed: 0"])
        xtest = xtest.drop(columns=["target", "id", "Unnamed: 0"])
        xval = xval.drop(columns=["target", "id", "Unnamed: 0"])
        features = xtrain.columns
        for feature in features:
            if xtrain[feature].std() == 0 and feature != "id":
                xtrain = xtrain.drop(columns=feature)
                xtest = xtest.drop(columns=feature)
                xval = xval.drop(columns=feature)
    else:
        """
        Gather features by relevance.
        """
        relevance_df = pd.read_csv(path_to_relevance)
        xtrain = xtrain[relevance_df["feature"][relevance_df["relevance"]>relevance]]
        xtest = xtest[relevance_df["feature"][relevance_df["relevance"]>relevance]]
        xval = xval[relevance_df["feature"][relevance_df["relevance"]>relevance]]
    nfeatures = xtrain.shape[1]
    if relevance < 0:
        accumulated_relevance = 1
    else:
        accumulated_relevance = np.sum(relevance_df["relevance"][relevance_df["relevance"]>relevance])

    return xtrain, xval, xtest, ytrain, yval, ytest, nfeatures, accumulated_relevance

def transform_data(xtrain, ytrain, xtest, ytest, n_components=None, use_pca=False):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardized
    and low-dimensional train and test sets together with the scaler object for the
    target values.
    Arguments:
        xtrain: size=(ntrain, p),
            training input
        ytrain: size=(ntrain, ?),
            training truth, ? depends on what we train against (mulitple objective)
        xtest: size=(ntest, p),
            testing input
        ytest: size=(ntest, ?),
            testing truth
        n_components: int,
            number of principal components used if use_pca=True
        use_pca: bool,
            if true use principal component analysis for dimensionality reduction

    Returns:
        xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled, yscaler
    """

    xscaler = StandardScaler()
    xtrain_scaled = xscaler.fit_transform(xtrain)
    xtest_scaled = xscaler.transform(xtest)
    yscaler = StandardScaler()
    ytrain_scaled = yscaler.fit_transform(ytrain)
    ytest_scaled = yscaler.transform(ytest)

    if use_pca:
        pca = PCA(n_components)
        xtrain_scaled = pca.fit_transform(xtrain)
        print("Fraction of variance retained is: ", np.sum(pca.explained_variance_ratio_))
        xtest_scaled = pca.transform(xtest)
    return xtrain_scaled, xtest_scaled, ytrain_scaled, ytest_scaled, yscaler

def load_data(data_path, target_path, target):
    df_target = pd.read_csv(target_path)
    df = pd.read_csv(data_path)
    target_vector = []
    for i, name1 in enumerate(df["id"]):
        for j, name2 in enumerate(df_target["id"]):
            if name1 == name2:
                target_vector.append(df_target[target][j])
    df = df.dropna(axis=1)
    return df, target_vector


def load_data_cv(data_path, target_path, target, depth_ac, type_GBM):

    print("Loading data... data-preprocess")
    print("Data path: ", data_path, " Target path: ", target_path, " Target: ", target, " Depth: ", depth_ac, " type: ", type_GBM)
    
    df = pd.read_csv(data_path)

    # load features
    feature_M_Full = load_mof_features(depth_ac, "M", "Full")
    feature_M_Conn = load_mof_features(depth_ac, "M", "Conn")
    feature_M_Func = load_mof_features(depth_ac, "M", "Func")

    feature_S_Full = load_mof_features(depth_ac, "S", "Full")
    feature_S_Conn = load_mof_features(depth_ac, "S", "Conn")
    feature_S_Func = load_mof_features(depth_ac, "S", "Func")

    # unpack the feature labels
    atom_based_features_M_Full, bond_based_features_M_Full, \
    atom_bond_based_features_M_full, sbu_atom_based_features_M, sbu_bond_based_features_M, \
    geometric_features  = feature_M_Full 

    atom_based_features_M_Conn, bond_based_features_M_Conn, \
    atom_bond_based_features_M_Conn, sbu_atom_based_features_M, sbu_bond_based_features_M, \
    geometric_features = feature_M_Conn

    atom_based_features_M_Func, bond_based_features_M_Func, \
    atom_bond_based_features_M_Func, sbu_atom_based_features_M, sbu_bond_based_features_M, \
    geometric_features  = feature_M_Func

    atom_based_features_S_Full, bond_based_features_S_Full, \
    atom_bond_based_features_S_full, sbu_atom_based_features_S, sbu_bond_based_features_S, \
    geometric_features  = feature_S_Full

    atom_based_features_S_Conn, bond_based_features_S_Conn, \
    atom_bond_based_features_S_Conn, sbu_atom_based_features_S, sbu_bond_based_features_S, \
    geometric_features = feature_S_Conn

    atom_based_features_S_Func, bond_based_features_S_Func, \
    atom_bond_based_features_S_Func, sbu_atom_based_features_S, sbu_bond_based_features_S, \
    geometric_features = feature_S_Func


    # select the features and add the target column
    # add the target column with the rest of the features
    # select the features
    if type_GBM == 'MS':    # both multiplication and substraction
        df = df[
            atom_based_features_M_Full + atom_based_features_M_Conn + atom_based_features_M_Func + \
            atom_based_features_S_Full + atom_based_features_S_Conn + atom_based_features_S_Func + \
            bond_based_features_M_Full + bond_based_features_M_Conn + bond_based_features_M_Func + \
            bond_based_features_S_Full + bond_based_features_S_Conn + bond_based_features_S_Func + \
            atom_bond_based_features_M_full + atom_bond_based_features_M_Conn + atom_bond_based_features_M_Func + \
            atom_bond_based_features_S_full + atom_bond_based_features_S_Conn + atom_bond_based_features_S_Func + \
            sbu_atom_based_features_M + sbu_bond_based_features_M + \
            sbu_atom_based_features_S +  sbu_bond_based_features_S +  \
            ["id"]]
    
    elif type_GBM == 'M':   # multiplication
        df = df[
            atom_based_features_M_Full + atom_based_features_M_Conn + atom_based_features_M_Func + \
            bond_based_features_M_Full + bond_based_features_M_Conn + bond_based_features_M_Func + \
            atom_bond_based_features_M_full + atom_bond_based_features_M_Conn + atom_bond_based_features_M_Func + \
            sbu_atom_based_features_M + sbu_bond_based_features_M + \
            ["id"]]

    elif type_GBM == 'S':    # substraction
        df = df[
            atom_based_features_S_Full + atom_based_features_S_Conn + atom_based_features_S_Func + \
            bond_based_features_S_Full + bond_based_features_S_Conn + bond_based_features_S_Func + \
            atom_bond_based_features_S_full + atom_bond_based_features_S_Conn + atom_bond_based_features_S_Func + \
            sbu_atom_based_features_S +  sbu_bond_based_features_S +  \
            ["id"]]

    elif type_GBM == 'Linker_MS':   # linker multiplication and substraction
        df = df[
            atom_based_features_M_Full + atom_based_features_M_Conn + atom_based_features_M_Func + \
            atom_based_features_S_Full + atom_based_features_S_Conn + atom_based_features_S_Func + \
            bond_based_features_M_Full + bond_based_features_M_Conn + bond_based_features_M_Func + \
            bond_based_features_S_Full + bond_based_features_S_Conn + bond_based_features_S_Func + \
            atom_bond_based_features_M_full + atom_bond_based_features_M_Conn + atom_bond_based_features_M_Func + \
            atom_bond_based_features_S_full + atom_bond_based_features_S_Conn + atom_bond_based_features_S_Func + \
            ["id"]]
       
    elif type_GBM == 'Linker_M':    # linker multiplication
        df = df[
            atom_based_features_M_Full + atom_based_features_M_Conn + atom_based_features_M_Func + \
            bond_based_features_M_Full + bond_based_features_M_Conn + bond_based_features_M_Func + \
            atom_bond_based_features_M_full + atom_bond_based_features_M_Conn + atom_bond_based_features_M_Func + \
            ["id"]]
        
    elif type_GBM == 'Linker_S':    # linker substraction
        df = df[
            atom_based_features_S_Full + atom_based_features_S_Conn + atom_based_features_S_Func + \
            bond_based_features_S_Full + bond_based_features_S_Conn + bond_based_features_S_Func + \
            atom_bond_based_features_S_full + atom_bond_based_features_S_Conn + atom_bond_based_features_S_Func + \
            ["id"]]

    if type_GBM == 'SBU_MS':    # both multiplication and substraction
        df = df[
            sbu_atom_based_features_M + sbu_bond_based_features_M + \
            sbu_atom_based_features_S +  sbu_bond_based_features_S +  \
            ["id"]]

    if type_GBM == 'SBU_M':    # multiplication
        df = df[
            sbu_atom_based_features_M + sbu_bond_based_features_M + \
            ["id"]]

    if type_GBM == 'SBU_S':    # substraction
        df = df[
            sbu_atom_based_features_S + sbu_bond_based_features_S + \
            ["id"]]

    # the version with geometry!!!
    elif type_GBM == 'M_geom':   # multiplication with geometry
        df = df[
            atom_based_features_M_Full + atom_based_features_M_Conn + atom_based_features_M_Func + \
            bond_based_features_M_Full + bond_based_features_M_Conn + bond_based_features_M_Func + \
            atom_bond_based_features_M_full + atom_bond_based_features_M_Conn + atom_bond_based_features_M_Func + \
            sbu_atom_based_features_M + sbu_bond_based_features_M + geometric_features + \
            ["id"]]

    if type_GBM == 'MS_geom':    # both multiplication and substraction with geometry
        df = df[
            atom_based_features_M_Full + atom_based_features_M_Conn + atom_based_features_M_Func + \
            atom_based_features_S_Full + atom_based_features_S_Conn + atom_based_features_S_Func + \
            bond_based_features_M_Full + bond_based_features_M_Conn + bond_based_features_M_Func + \
            bond_based_features_S_Full + bond_based_features_S_Conn + bond_based_features_S_Func + \
            atom_bond_based_features_M_full + atom_bond_based_features_M_Conn + atom_bond_based_features_M_Func + \
            atom_bond_based_features_S_full + atom_bond_based_features_S_Conn + atom_bond_based_features_S_Func + \
            sbu_atom_based_features_M + sbu_bond_based_features_M + \
            sbu_atom_based_features_S +  sbu_bond_based_features_S + geometric_features + \
            ["id"]]
        
   
    # print columns of the dataframe
    print("Columns of the dataframe: ", df.columns)

    df_target = pd.read_csv(target_path)

    target_vector = []
    for i, name1 in enumerate(df["id"]):
        for j, name2 in enumerate(df_target["id"]):
            if name1 == name2:
                target_vector.append(df_target[target][j])

    removals = ["id"]
    df = df.drop(columns=removals)
    df = df.dropna(axis=1)
    df["target"] = target_vector
    print("Data loaded", df.shape, df)

    # check how many 'outliers', columns with the same value across it exist
    outlier = []

    # Check standard deviation for each column
    for col in df.columns:
        col_std = df[col].std()  # Calculate std separately
        
        if col_std.item() == 0:
            #print(f"Column {col} has a standard deviation of zero.")
            outlier.append(col)
    print('Same column values', len(outlier), outlier)
    df = df.drop(outlier, axis=1)


    return df     

def scale_features(xtrain, xtest):
    
    xscaler = StandardScaler()
    xtrain_scaled = xscaler.fit_transform(xtrain)
    xtest_scaled = xscaler.transform(xtest)
    return xtrain_scaled, xtest_scaled, xscaler


