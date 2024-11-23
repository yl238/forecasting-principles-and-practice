import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdatasets
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from plotnine import aes, geom_line, ggplot, labs, theme
from sklearn import metrics

sns.set(font_scale=1.0)
plt.rcParams["lines.linewidth"] = 1.0


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


def summarize_ets(fitted):
    output = pd.DataFrame(
        dict(
            actual=fitted.data.orig_endog,
            level=fitted.level,
            trend=fitted.trend,
            season=fitted.season,
            forecast=fitted.fittedvalues,
            remainder=fitted.resid,
        )
    )
    return output


def ciclean(ci_df):
    """Clean up conf_int() result column names."""
    ci_df = ci_df.copy()
    ci_df.columns = "lower", "upper"
    return ci_df


def model_evaluation(y_true, y_pred, Model):

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"Model Evaluation: {Model}")
    print(f"MSE is : {metrics.mean_squared_error(y_true, y_pred)}")
    print(f"MAE is : {metrics.mean_absolute_error(y_true, y_pred)}")
    print(f"RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}")
    print(f"MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}")
    print(f"R2 is : {metrics.r2_score(y_true, y_pred)}")
    print(f"corr is : {np.corrcoef(y_true, y_pred)[0,1]}", end="\n\n")
