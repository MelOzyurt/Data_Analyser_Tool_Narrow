import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind

# ---------- Load Data ----------
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

# ---------- 1. Numeric Summary ----------
def analyze_numeric(df):
    numeric_df = df.select_dtypes(include=np.number)
    summary = numeric_df.describe().transpose()
    return summary

# ---------- 2. Correlation Plot ----------
def correlation_plot(df):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix')
    return fig, corr

# ---------- 3. Chi-Square Test ----------
def chi_square_analysis(df, col1=None, col2=None):
    if col1 is None or col2 is None:
        # Just return column names so user can pick
        return {"error": "You must specify two categorical columns."}, None

    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    results = pd.DataFrame({
        "Chi² Statistic": [chi2],
        "Degrees of Freedom": [dof],
        "P-Value": [p]
    })

    return results, p

#-------------4. T-Test Analaysis-----------------------



from scipy.stats import ttest_ind
import pandas as pd

def t_test_analysis(df, col1, col2):
    try:
        group1 = df[col1].dropna()
        group2 = df[col2].dropna()
        stat, p_val = ttest_ind(group1, group2, equal_var=False)

        result = pd.DataFrame({
            "Group": [col1, col2],
            "Mean": [group1.mean(), group2.mean()],
            "Std": [group1.std(), group2.std()],
            "Count": [len(group1), len(group2)]
        })

        return result, p_val
    except Exception as e:
        return f"Hata: {e}", None

 
