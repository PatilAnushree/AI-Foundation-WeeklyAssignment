import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data + Probability Analyzer", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Preprocessing",
    "Statistics Dashboard",
    "Probability Calculator",
    "Simulation"
])

# Store dataset globally
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------------
# 1️⃣ Upload & Preprocessing
# -------------------------------
if page == "Upload & Preprocessing":
    st.title("📂 Upload & Preprocessing")

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.session_state.df = df

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write("### Dataset Info")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(df.isnull().sum())

        st.subheader("Handle Missing Values")
        option = st.selectbox("Choose method", ["None", "Drop", "Fill Mean", "Fill Median"])

        if option == "Drop":
            df = df.dropna()
        elif option == "Fill Mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif option == "Fill Median":
            df = df.fillna(df.median(numeric_only=True))

        st.session_state.df = df

# -------------------------------
# 2️⃣ Statistics Dashboard
# -------------------------------
elif page == "Statistics Dashboard":
    st.title("📊 Statistics Dashboard")

    df = st.session_state.df

    if df is None:
        st.warning("⚠️ Please upload dataset first!")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols)

            st.write("### Statistical Values")
            st.write("Mean:", df[col].mean())
            st.write("Median:", df[col].median())
            st.write("Mode:", df[col].mode()[0])
            st.write("Variance:", df[col].var())
            st.write("Standard Deviation:", df[col].std())
            st.write("Min:", df[col].min())
            st.write("Max:", df[col].max())

            # Histogram
            st.write("### Histogram")
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20)
            st.pyplot(fig)

            # Boxplot
            st.write("### Boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col])
            st.pyplot(fig)

# -------------------------------
# 3️⃣ Probability Calculator
# -------------------------------
elif page == "Probability Calculator":
    st.title("📈 Probability Calculator")

    df = st.session_state.df

    if df is None:
        st.warning("⚠️ Please upload dataset first!")
    else:
        col = st.selectbox("Select column", df.columns)
        value = st.text_input("Enter value")

        if value:
            try:
                value = float(value)
            except:
                pass

            prob = len(df[df[col] == value]) / len(df)
            st.write(f"Probability P({col} = {value}) = {prob}")

# -------------------------------
# 4️⃣ Simulation Module
# -------------------------------
elif page == "Simulation":
    st.title("🎲 Simulation Module")

    sim_type = st.selectbox("Choose simulation", ["Coin Toss", "Dice Roll"])
    trials = st.slider("Number of trials", 10, 1000, 100)

    if st.button("Run Simulation"):
        if sim_type == "Coin Toss":
            results = np.random.choice(["H", "T"], size=trials)
        else:
            results = np.random.randint(1, 7, size=trials)

        unique, counts = np.unique(results, return_counts=True)
        freq = dict(zip(unique, counts))

        st.write("### Frequency Distribution")
        st.write(freq)

        fig, ax = plt.subplots()
        ax.bar(freq.keys(), freq.values())
        st.pyplot(fig)

        st.write("### Experimental Probability")
        for k, v in freq.items():
            st.write(f"P({k}) = {v/trials}")

        st.write("### Theoretical Probability")
        if sim_type == "Coin Toss":
            st.write("P(H) = 0.5, P(T) = 0.5")
        else:
            st.write("Each outcome = 1/6")