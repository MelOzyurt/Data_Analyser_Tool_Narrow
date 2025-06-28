# utils_text.py

def interpret_correlation(corr_df):
    """
    Correlation matrix yorumlamak için basit metin.
    Pozitif ve negatif korelasyonları açıklıyor.
    """
    if corr_df.empty:
        return "No numeric columns available for correlation."

    strong_corrs = []
    for col1 in corr_df.columns:
        for col2 in corr_df.columns:
            if col1 != col2:
                val = corr_df.loc[col1, col2]
                if abs(val) > 0.7:
                    relation = "positive" if val > 0 else "negative"
                    strong_corrs.append(f"Strong {relation} correlation between {col1} and {col2}: {val:.2f}")

    if strong_corrs:
        return "\n".join(strong_corrs)
    else:
        return "No strong correlations found (>|0.7|)."

def interpret_chi_square(p_val):
    """
    Chi-square p-değerine göre basit yorum döner.
    """
    if p_val is None:
        return "No p-value provided."
    elif p_val < 0.05:
        return "There is a significant association between the variables (p < 0.05)."
    else:
        return "No significant association found between the variables (p >= 0.05)."
