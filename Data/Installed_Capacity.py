# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 08:52:52 2022

@author: marti
"""


import pandas as pd
import numpy as np


if __name__ == "__main__":
    units_full = pd.read_csv("units_full_info.csv")
    units_full_DK1_1 = units_full.loc[units_full["UtilizationFactor"] == "DK1_PW_01_UF"]
    CAP = units_full_DK1_1.loc[:, "INST_CAP_KW"].to_numpy()
    print(np.sum(CAP))
