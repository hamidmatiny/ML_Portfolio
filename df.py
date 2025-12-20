import pandas as pd
data = pd.read_csv("/Users/hamidrezamatiny/Downloads/titanic/train.csv")
data["Age"] = data["Age"].fillna(data["Age"].median())
print(data["Age"].isnull().sum())