import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, sympify
import time

# ======================
# 🌙 DARK MODE CONFIG
# ======================
st.set_page_config(page_title="ML Visualizer", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stApp {
    background-color: #0E1117;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 📌 NAVBAR
# ======================
menu = st.sidebar.radio("📌 Navigation", [
    "Vector Similarity (3D)",
    "Function & Tangent",
    "Gradient Descent"
])

st.title("🚀 ML Foundations Visualizer")

# ======================
# 📐 MODULE 1: 3D VECTOR
# ======================
if menu == "Vector Similarity (3D)":
    st.header("📐 3D Vector Visualization")

    v1 = st.text_input("Vector 1", "1,2,3")
    v2 = st.text_input("Vector 2", "4,5,6")

    v1 = np.array(list(map(float, v1.split(","))))
    v2 = np.array(list(map(float, v2.split(","))))

    dot = np.dot(v1, v2)
    cos = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
    dist = np.linalg.norm(v1 - v2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dot Product", round(dot, 3))
    col2.metric("Cosine Similarity", round(cos, 3))
    col3.metric("Distance", round(dist, 3))

    # 3D Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
        mode='lines+markers',
        name='Vector 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]],
        mode='lines+markers',
        name='Vector 2'
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    st.plotly_chart(fig, use_container_width=True)

# ======================
# 📈 MODULE 2: TANGENT
# ======================
elif menu == "Function & Tangent":
    st.header("📈 Tangent Line Animation")

    x = symbols('x')
    func_input = st.text_input("Enter function", "x**2")

    try:
        func = sympify(func_input)
        derivative = diff(func, x)

        f = lambdify(x, func, "numpy")
        d = lambdify(x, derivative, "numpy")

        point = st.slider("Select point for tangent", -10.0, 10.0, 1.0)

        x_vals = np.linspace(-10, 10, 200)
        y_vals = f(x_vals)

        slope = d(point)
        tangent = slope * (x_vals - point) + f(point)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="Function")
        ax.plot(x_vals, tangent, '--', label="Tangent")
        ax.scatter(point, f(point), color='red')
        ax.legend()

        st.pyplot(fig)

        # Animation
        if st.button("▶ Animate Tangent"):
            for p in np.linspace(-5, 5, 30):
                slope = d(p)
                tangent = slope * (x_vals - p) + f(p)

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals)
                ax.plot(x_vals, tangent, '--')
                ax.scatter(p, f(p), color='red')

                st.pyplot(fig)
                time.sleep(0.2)

    except:
        st.error("Invalid function!")

# ======================
# 📉 MODULE 3: GRADIENT DESCENT
# ======================
elif menu == "Gradient Descent":
    st.header("📉 Gradient Descent (Multiple Functions)")

    func_choice = st.selectbox("Choose Function", [
        "x^2",
        "x^3 - 3x",
        "sin(x)"
    ])

    lr = st.slider("Learning Rate", 0.001, 1.0, 0.1)
    iterations = st.slider("Iterations", 10, 200, 50)
    start = st.number_input("Start Point", value=5.0)

    # Functions
    if func_choice == "x^2":
        func = lambda x: x**2
        grad = lambda x: 2*x

    elif func_choice == "x^3 - 3x":
        func = lambda x: x**3 - 3*x
        grad = lambda x: 3*x**2 - 3

    elif func_choice == "sin(x)":
        func = lambda x: np.sin(x)
        grad = lambda x: np.cos(x)

    x = start
    x_hist = []
    y_hist = []

    for i in range(iterations):
        x_hist.append(x)
        y_hist.append(func(x))
        x = x - lr * grad(x)

    st.success(f"Final Value: {round(x, 4)}")

    # Plot Function
    x_vals = np.linspace(-10, 10, 200)
    y_vals = func(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals)
    ax.scatter(x_hist, y_hist, color='red')
    ax.set_title("Optimization Path")

    st.pyplot(fig)

    # Loss Graph
    fig2, ax2 = plt.subplots()
    ax2.plot(y_hist)
    ax2.set_title("Loss vs Iterations")

    st.pyplot(fig2)