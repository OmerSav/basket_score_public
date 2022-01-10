# -*- coding: utf-8 -*-
#  Simple proccess and merge scraped data files and save single file.
import os

import pandas as pd

data_path = "./data"
files = os.listdir(data_path)

dataframes = []
for elm in files:
    dataframes.append(pd.read_csv("./data/" + elm, index_col=0))

df = pd.concat(dataframes)
df = df.dropna()
df = df.sort_values(by=['date'], ascending=False)
df = df.reset_index(drop=True)
df["month"] = df["date"].apply(lambda a: int(a[5:7]))

df.to_csv("score_data.csv")
