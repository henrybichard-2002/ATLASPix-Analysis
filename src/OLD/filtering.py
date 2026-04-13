# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:40:04 2025

@author: henry
"""

import pandas as pd
from typing import Optional

def filter_tot(
    df: pd.DataFrame,
    min_tot: Optional[float] = None,
    max_tot: Optional[float] = None
) -> pd.DataFrame:
    if min_tot is not None and max_tot is not None and min_tot > max_tot:
        raise ValueError("min_tot cannot be greater than max_tot.")
    mask = pd.Series(True, index=df.index)
    if min_tot is not None:
        mask &= (df['ToT'] >= min_tot)
    if max_tot is not None:
        mask &= (df['ToT'] <= max_tot)
    return df[mask]