import pandas as pd
import pathlib

good_fit_data = pd.read_excel(
    pathlib.Path("data/good_fit_data.xlsx"), sheet_name="adjusted sample sensogram"
)
print(good_fit_data.head())
