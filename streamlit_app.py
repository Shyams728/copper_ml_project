
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib

# Import utility functions
import model_utils as mu

# Page Configuration
st.set_page_config(
    page_title="Copper Industry Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
    }
    .card {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Data (Cached)
@st.cache_data
def load_data():
    DATA_PATH = 'model/cleaned_copper_data.csv'
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Convert dates
            for col in ['item_date', 'delivery_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

# Load Models
artifacts = mu.load_artifacts()
missing_models = []
if artifacts.get('reg_model') is None:
    missing_models.append("Regression model")
if artifacts.get('cls_model') is None:
    missing_models.append("Classification model")

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/copper-ore.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to:", ["Home", "EDA Dashboard", "Price Prediction", "Lead Status Classification"])
    
    st.markdown("---")
    st.info("Industrial Copper Modeling Project\nPredicting Pricing and Lead Status")
    if missing_models:
        st.warning(
            "Model artifacts not found. Run `python prepare_data.py` and "
            "`python train_models.py` to generate the required files."
        )

# --- HOME PAGE ---
if page == "Home":
    st.markdown("<div class='main-header'>Industrial Copper Modeling</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Project Overview
        This machine learning application serves the copper industry by providing advanced analytics for pricing and sales.
        
        **Key Features:**
        - **Selling Price Prediction**: Utilize regression models to estimate the optimal selling price for copper products.
        - **Lead Outcome Classification**: Predict whether a sales lead will be 'Won' or 'Lost' to prioritize efforts.
        - **Data Insights**: Interactive dashboard to explore historical transaction data.
        """)
    
    with col2:
        st.image("shubham-dhage-AC4Q1uLRKd4-unsplash.jpg", 
                 caption="Copper Production", use_column_width=True)

    st.markdown("### 🔧 How it Works")
    st.code("""
    1. Upload/Load Data -> Clean & Preprocess
    2. Regression Model -> Predict Selling Price ($)
    3. Classification Model -> Predict Lead Status (Won/Lost)
    """)

# --- EDA DASHBOARD ---
elif page == "EDA Dashboard":
    st.markdown("<div class='main-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Total Volume (Tons)", f"{df['quantity_tons'].sum():,.0f}")
        col3.metric("Avg Selling Price", f"${df['selling_price'].mean():,.2f}")
        
        if 'leads' in df.columns:
            won_pct = (df['leads'] == 'Won').mean() * 100
            col4.metric("Win Rate", f"{won_pct:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Price Analysis", "Volume Analysis", "Categorical Distribution"])
        
        with tab1:
            st.subheader("Selling Price Distribution")
            fig = px.histogram(df, x="selling_price", nbins=100, title="Distribution of Selling Price",
                               color_discrete_sequence=['#1E88E5'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Price vs. Quantity")
            sample_df = df.sample(min(5000, len(df))) # Sample for performance
            fig2 = px.scatter(sample_df, x="quantity_tons", y="selling_price", 
                              color="item_type", title="Selling Price vs Quantity (Sampled)",
                              log_x=True, log_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("Quantity Distribution by Application")
            fig3 = px.box(df, x="application", y="quantity_tons", title="Quantity Distribution by Application",
                          log_y=True)
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            col_cat1, col_cat2 = st.columns(2)
            with col_cat1:
                st.subheader("Item Type Distribution")
                fig4 = px.pie(df, names="item_type", title="Transactions by Item Type")
                st.plotly_chart(fig4, use_container_width=True)
            
            with col_cat2:
                if 'leads' in df.columns:
                    st.subheader("Lead Status Distribution")
                    fig5 = px.bar(df['leads'].value_counts().reset_index(), x='index', y='leads', 
                                  title="Lead Outcome Counts")
                    st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning(
            "Data not found. Run `python prepare_data.py` to generate "
            "`model/cleaned_copper_data.csv` or upload data."
        )
        uploaded_file = st.file_uploader("Upload Copper Data (Excel)", type="xlsx")
        if uploaded_file:
            # Logic to handle upload could go here (omitted for brevity)
            pass

# --- PREDICTION PAGE (Common Inputs) ---
elif page in ["Price Prediction", "Lead Status Classification"]:
    task_name = "Predicting Selling Price" if page == "Price Prediction" else "Predicting Lead Status"
    st.markdown(f"<div class='main-header'>{task_name}</div>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.subheader("Enter Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quantity = st.number_input("Quantity (Tons)", min_value=0.01, value=10.0, step=0.1)
            thickness = st.number_input("Thickness", min_value=0.01, value=1.0, step=0.1)
            width = st.number_input("Width", min_value=1.0, value=1000.0, step=10.0)
            
        with col2:
            customer = st.text_input("Customer Code", value="30156308")
            country = st.selectbox("Country Code", ["28", "25", "30", "32", "38", "78", "27", "77", "113", "79", "26", "39", "78", "84", "80"])
            item_type = st.selectbox("Item Type", ["W", "WI", "S", "Others", "PL", "IPL", "SLAWR"])
            
        with col3:
            application = st.text_input("Application Code", value="10")
            product_ref = st.text_input("Product Ref", value="1670798778")
            item_date = st.date_input("Item Date", datetime.now())
            delivery_date = st.date_input("Delivery Date", datetime.now())
            
        submit = st.form_submit_button("Predict")
    
    if submit:
        if missing_models:
            st.error("Missing model artifacts. Run `python prepare_data.py` and `python train_models.py`.")
            st.stop()
        # Prepare input data
        # Note: We need to handle year/month/day extraction if that was done in training
        # Assuming model_utils/preprocessor handles raw dates or we do it here.
        # Based on copper_modeling_final.py, it created: _year, _month, _quarter, _dayofweek
        
        input_data = {
            'quantity_tons': quantity,
            'customer_code': customer, # Renamed in cleaning
            'country_code': country,
            'item_type': item_type,
            'application': application,
            'thickness': thickness,
            'width': width,
            'product_ref': product_ref,
            'item_date': pd.to_datetime(item_date),
            'delivery_date': pd.to_datetime(delivery_date),
            # Add calculated date features if the model expects them explicitly and preprocessor doesn't genericize
        }
        
        # Temporal features manual creation (if model expects them explicitly before pipeline)
        # Looking at copper_modeling_final.py: FeatureEngineer.create_temporal_features does this.
        # But we need to match the feature names EXACTLY.
        for date_col in ['item_date', 'delivery_date']:
            dt = input_data[date_col]
            input_data[f'{date_col}_year'] = dt.year
            input_data[f'{date_col}_month'] = dt.month
            input_data[f'{date_col}_quarter'] = (dt.month - 1) // 3 + 1
            input_data[f'{date_col}_dayofweek'] = dt.dayofweek
            # Original script might drop the original date columns?
            # DataCleaner converts them, FeatureEngineer extracts, then DataPreparer might drop them.
        
        # We assume 'id', 'material_ref' are not needed or were dropped.
        
        if page == "Price Prediction":
            with st.spinner("Calculating Price..."):
                prediction = mu.predict_selling_price(input_data, artifacts)
                
            if prediction is not None:
                st.success(f"### Predicted Selling Price: ${prediction:,.2f}")
                st.info("Note: This is an estimated price based on historical data.")
            else:
                 st.error("Prediction failed. Check logs.")
                
        else: # Lead Status
             with st.spinner("Analyzing Lead..."):
                pred_class, probabilities = mu.predict_status(input_data, artifacts)
                
             if pred_class is not None:
                 status = "WON" if pred_class == 1 else "LOST"
                 color = "green" if status == "WON" else "red"
                 st.markdown(f"### Predicted Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
                 
                 if probabilities is not None:
                     # prob[1] is probability of class 1 (Won)
                     confidence = probabilities[1] if pred_class == 1 else probabilities[0]
                     st.progress(float(confidence))
                     st.caption(f"Confidence: {confidence*100:.1f}%")

st.markdown("---")
st.caption("© 2024 Industrial Copper Analytics | v1.0")
