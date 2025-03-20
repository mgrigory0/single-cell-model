#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from pycytominer import normalize




## Axiom data of 4 compounds
final_data= pd.read_csv('/Users/grigoryanmariam/Desktop/2rep_4cpds.csv')



features_list = pd.read_csv("/Users/grigoryanmariam/Library/Mobile Documents/com~apple~CloudDocs/Documents/thèse/featset_002.csv")



features = features_list["feature"].tolist()


# without metadata
columns_to_drop = ['TableNumber', 'ImageNumber', 'ObjectNumber', 'FileName_CellOutlines', 'well', 'name', 'plate', 'replicate', 'concentration']  
fd= final_data.drop(columns=columns_to_drop)




## Normalization per plate 



unique_plates = final_data['plate'].unique()

normalized_dfs = []


for plate in unique_plates:

    plate_data = final_data[final_data['plate'] == plate]
    print(plate)
    print(plate_data['name'].unique())
   
    normalized_plate = normalize(
        profiles=plate_data,
        features="infer",  
        image_features=False,
        meta_features=['TableNumber', 'ImageNumber', 'ObjectNumber', 'name', 'plate', 'replicate', 'well', 'concentration'],
        samples="name == 'DMSO' ",  
        method="mad_robustize"
    )
    

    normalized_dfs.append(normalized_plate)


normalized_df = pd.concat(normalized_dfs, ignore_index=True)



## feature selection by Jump features


jump_data = normalized_df[features]



## feature selection manuelly, I got 1439 features

def remove_low_variance_features(df, threshold):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    return df[df.columns[selector.get_support(indices=True)]]

def remove_correlated_features(df, correlation_threshold=0.95):
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    return df.drop(columns=to_drop)

def remove_redundant_features(df):
    return df.loc[:, ~df.columns.duplicated()]


# without metadata
columns_to_drop = ['TableNumber', 'ImageNumber', 'ObjectNumber', 'well', 'name', 'plate', 'replicate', 'concentration']  
fd= normalized_df.drop(columns=columns_to_drop)



normalized_df2 = remove_low_variance_features(fd, threshold=0.01)
normalized_df2 = remove_correlated_features(normalized_df2, correlation_threshold=0.95)
normalized_df2 = remove_redundant_features(normalized_df2)

print(normalized_df2.head())



## by a PCA 





