
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    norm, gamma, weibull_min, expon, lognorm, beta,
    uniform, triang, chi2, pareto, laplace)
# ----------------------------------------
# Global Variable/Function Center
# ----------------------------------------

distribution_options = {
    "Normal": norm,
    "Gamma": gamma,
    "Weibull": weibull_min,
    "Exponential": expon,
    "Lognormal": lognorm,
    "Beta": beta,
    "Uniform": uniform,
    "Triangular": triang,
    "Chi-Square": chi2,
    "Pareto": pareto,
    "Laplace": laplace
}

PARAM_SLIDERS = {
    "Normal": {
        "shape": [],
        "loc": ("Mean (μ)", -10.0, 10.0, 0.0),
        "scale": ("Std Dev (σ)", 0.1, 5.0, 1.0)
    },

    "Gamma": {
        "shape": [
            ("Shape (k)", 0.1, 10.0, 2.0)
        ],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("Scale (θ)", 0.1, 5.0, 1.0)
    },

    "Weibull": {
        "shape": [
            ("Shape (k)", 0.1, 5.0, 1.5)
        ],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("Scale (λ)", 0.1, 5.0, 1.0)
    },

    "Exponential": {
        "shape": [],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("Scale", 0.1, 5.0, 1.0)
    },

    "Lognormal": {
        "shape": [
            ("Shape (σ)", 0.1, 2.0, 0.5)
        ],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("Scale", 0.1, 5.0, 1.0)
    },

    "Beta": {
        "shape": [
            ("Alpha (a)", 0.1, 5.0, 2.0),
            ("Beta (b)", 0.1, 5.0, 2.0)
        ],
        "loc": ("loc", -2.0, 2.0, 0.0),
        "scale": ("scale", 0.1, 5.0, 1.0)
    },

    "Uniform": {
        "shape": [],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("Width", 0.1, 10.0, 5.0)
    },

    "Triangular": {
        "shape": [
            ("Shape (c) 0→1", 0.0, 1.0, 0.5)
        ],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("scale", 0.1, 10.0, 5.0)
    },

    "Chi-Square": {
        "shape": [
            ("Degrees of Freedom", 1.0, 10.0, 3.0)
        ],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("scale", 0.1, 5.0, 1.0)
    },

    "Pareto": {
        "shape": [
            ("Shape (b)", 0.5, 5.0, 2.5)
        ],
        "loc": ("loc", -5.0, 5.0, 0.0),
        "scale": ("scale", 0.1, 5.0, 1.0)
    },

    "Laplace": {
        "shape": [],
        "loc": ("Mean", -5.0, 5.0, 0.0),
        "scale": ("Diversity (b)", 0.1, 5.0, 1.0)
    }
}
data_show = None
data = []
# ----------------------------------------
# Title
# ----------------------------------------

st.title("Interactive Data Distribution Visualizer")

st.divider()


input_col, display_col = st.columns(2)

# ----------------------------------------
# Input Column & Data Input
# ----------------------------------------

with input_col:
    st.header("Data Formatting Panel")

# File Upload

    mode = st.radio(
        "Select mode",
        ["Data Input", "Data Upload"],
        horizontal=True)

    st.write("Selected mode:", mode)

    if mode == "Data Input":
        with st.expander("Manual Data Input"):
            # Initialize the data list in session state
            if "data" not in st.session_state:
                st.session_state.data = []
    
            # Input for a new value
            new_value = st.text_input("Enter a number:")
    
            # Button to append the value
            if st.button("Add to data"):
                try:
                    value = float(new_value)
                    st.session_state.data.append(value)
                    st.success(f"Added {value}")
                except ValueError:
                    st.error("Please enter a valid number.")
            data = st.session_state.data

    
            # Show the current data array
            st.write("Current data array:", st.session_state.data)
    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        data = None

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:", df.head())
            data = df.iloc[:, 0].to_numpy(dtype=float)
    
#    ''' This is a Divider of Data Input and Distribution Parameters'''
    st.divider()
#    ''' This is a Divider of Data Input and Distribution Parameters'''

# ----------------------------------------
# Distribution Options
# ----------------------------------------    
    
    input_col,fit_col = st.columns(2)
    
    with input_col:
        manual_mode = st.toggle("Manual Mode")
        st.write("Manual Calibration Chosen:", manual_mode)
        
    with fit_col:
        dist_choice = st.selectbox("Choose Distrubtion Type:", list(distribution_options.keys()))
        dist = distribution_options[dist_choice]
    # Use Functions to make it easier to read and debugg issues
    if manual_mode:
        with st.expander("Advanced Manual Distribution Settings"):
            data_show = True
            #Slider Configuration Variable
            slider_config = PARAM_SLIDERS[dist_choice]
            shape_values = []
            for (label, minv, maxv, default) in slider_config["shape"]:
                val = st.slider(label, minv, maxv, default)
                shape_values.append(val)

            # Read loc and scale
            loc_label, loc_min, loc_max, loc_default = slider_config["loc"]
            scale_label, scale_min, scale_max, scale_default = slider_config["scale"]

            loc = st.slider(loc_label, loc_min, loc_max, loc_default)
            scale = st.slider(scale_label, scale_min, scale_max, scale_default)
            data_show = True
    else:
        if data is not None:
            if data is not None and len(data) > 0:
                params_auto = dist.fit(data)
                if len(params_auto) == 2:  # no shape parameters
                    loc, scale = params_auto
                    shape_values = []
                    data_show = True
                else:
                    *shape_values, loc, scale = params_auto
                    data_show = True
            else:
                st.warning("Please upload valid numeric data for automatic fitting.")
                shape_values = []
                loc = 0
                scale = 1
                
        else:
            st.warning("Please upload data for automatic fitting.")
            shape_values = []
            loc = 0
            scale = 1





# ----------------------------------------
# Display Column
# ----------------------------------------
    
with display_col:
    if data_show == True:
        st.header("Data Visualization Panel")
        x = np.linspace(-10, 10, 400)
        params = [*shape_values, loc, scale]
    
        pdf = dist.pdf(x, *params)
        
        fig, ax = plt.subplots()
        ax.plot(x, pdf, linewidth=2)
        ax.set_title(f"{dist_choice} Distribution")
        st.pyplot(fig)
    else:
        st.header("Error Detected")



