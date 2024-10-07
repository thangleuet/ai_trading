import pandas as pd

df_raw = pd.read_csv(r"data/exness_xau_usd_m30.csv")
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_train = df_raw[df_raw['Date'].dt.year.isin([2020, 2021, 2022])]
df_test = df_raw[df_raw['Date'].dt.year.isin([2023])]

df_train.to_csv(r"train/exness_xau_usd_m30_2020_2021_2022_train.csv", index=False)
df_test.to_csv(r"test/exness_xau_usd_m30_2023_test.csv", index=False)