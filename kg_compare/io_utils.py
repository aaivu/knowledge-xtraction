from __future__ import annotations

import pandas as pd


def read_excel(path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def write_excel(df: pd.DataFrame, out_path: str, sheet_name: str = "triplet_analysis") -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
