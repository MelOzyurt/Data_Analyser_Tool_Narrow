#Merged app.py


# gpt-4.1 version


import streamlit as st
import pandas as pd
import numpy as np
import re
import openai
from analysis_utils import *
from utils_text import *
from analysis_utils import t_test_analysis

# ✅ Application configration
st.set_page_config(page_title="📊 Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer")

# OpenAI API Key
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# AI yorum fonksiyonu (tam cümle ve token sınırıyla)
def ai_interpretation(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes data and provides insights. You can highlight anomalies, interpret correlations between attributes, find and tell similarities or impact from other attributes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        raw_message = response.choices[0].message.content.strip()

        # Sadece tam cümleleri al
        sentences = re.findall(r'[^.!?]*[.!?]', raw_message)
        clean_message = ''.join(sentences).strip()

        return clean_message

    except Exception as e:
        return f"**Error during AI interpretation:** {e}"

# ✅ Uploading File
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, XML, Feather)",
    type=["csv", "xlsx", "xls", "json", "xml", "feather"]
)

if uploaded_file:
    try:
        # File Typles
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xml'):
            df = pd.read_xml(uploaded_file)
        elif uploaded_file.name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()


  # ✅ Data Priview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())



    option = st.selectbox("Select Analysis Type", [
        "Numeric Summary",
        "Correlation Matrix",
        "Chi-Square Test",
        "T-Test"
    ])

    if option == "Numeric Summary":
        result = analyze_numeric(df)
        st.write(result)

        # AI Yorumu
        prompt = f"Analyze the following numeric summary statistics and provide insights:\n{result.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)

 elif option == "Correlation Matrix":
    fig, corr_df = correlation_plot(df)
    
    # Grafiği genişlet
    fig.update_layout(
        width=1000,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    st.plotly_chart(fig)


        prompt = f"Explain the key points and findings from this correlation matrix:\n{corr_df.to_string()}"
        ai_result = ai_interpretation(prompt)
        st.markdown("### AI Insights")
        st.write(ai_result)


    elif option == "Chi-Square Test":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) >= 2:
            col1 = st.selectbox("Select first categorical column", categorical_cols)
            col2 = st.selectbox("Select second categorical column", categorical_cols, index=1)
            if col1 != col2:
                result, p_val = chi_square_analysis(df, col1, col2)
                st.write(result)
                prompt = f"Interpret the chi-square test result with p-value {p_val} between {col1} and {col2}."
                ai_result = ai_interpretation(prompt)
                st.markdown("### AI Insights")
                st.write(ai_result)
        else:
            st.error("Dataset does not have enough categorical columns for Chi-Square test.")

    elif option == "T-Test":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Select first numeric column", numeric_cols)
            col2 = st.selectbox("Select second numeric column", numeric_cols, index=1)
            if col1 != col2:
                try:
                    result, p_val = t_test_analysis(df, col1, col2)
                    st.write(result)
                    prompt = f"Interpret the t-test result with p-value {p_val} comparing {col1} and {col2}."
                    ai_result = ai_interpretation(prompt)
                    st.markdown("### AI Insights")
                    st.write(ai_result)
                except Exception as e:
                    st.error(f"T-Test Error: {e}")
        else:
            st.error("Dataset does not have enough numeric columns for T-Test.")
