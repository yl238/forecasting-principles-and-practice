import numpy as np
import pandas as pd


def summarize(gb, f):
    """Summarize grouped things."""
    return gb.apply(lambda x: pd.Series(f(x)))


def compute(df, f):
    """Compute new (or replacement) columns."""
    newdf = pd.DataFrame(f(df), index=df.index)
    dropcols = [col for col in newdf.columns if col in df.columns]
    if dropcols:
        df = df.drop(columns=dropcols)
    return df.join(newdf)


def set_freq(df, freq=None):
    """Set frequency of DateTimeIndex."""
    if freq is None:
        freq = pd.infer_freq(df.index)
    return df.asfreq(freq)


def RMSE(y_test, y_pred):
    return np.sqrt(np.mean((y_test - y_pred) ** 2))


def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))


def MAPE(y_test, y_pred):
    return 100 * np.mean(np.abs((y_test - y_pred) / y_test))


def compute_accuracy(y_test, models):
    """Gather some metrics for a few models."""
    metrics_fns = RMSE, MAE, MAPE
    return pd.DataFrame(
        {
            label: [
                f(y_test, model.predict(y_test.index.min(), y_test.index.max()))
                for f in (RMSE, MAE, MAPE)
            ]
            for (label, model) in models.items()
        },
        index=[f.__name__ for f in metrics_fns],
    ).T
