import pandas as pd


def read_excel(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")
