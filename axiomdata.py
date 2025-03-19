#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:53:11 2025

@author: grigoryanmariam
"""



import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt







## Axiom data of 4 compounds
final_data= pd.read_csv('/Users/grigoryanmariam/Library/Mobile Documents/com~apple~CloudDocs/Documents/thèse/plates_4cpds/2rep_4cpds.csv')



features_list = pd.read_csv("/Users/grigoryanmariam/Library/Mobile Documents/com~apple~CloudDocs/Documents/thèse/featset_002.csv")



features = features_list["feature"].tolist()



jump_data = final_data[features]







