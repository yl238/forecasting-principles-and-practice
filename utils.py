import pandas as pd

def summarize(gb, f):
    """Summarize grouped things."""
    return gb.apply(lambda x: pd.Series(f(x)))
