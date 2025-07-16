import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Pizza Delivery Time Predictor",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 2rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
    }
    .metric-card h4 {
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .metric-card p {
        color: #333333;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍕 Pizza Delivery Time Predictor</h1>', unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Load your trained model and scaler
        model = pickle.load(open('best_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        
        # Load feature names untuk referensi
        try:
            feature_names = pd.read_csv('feature_names.csv')['feature_names'].tolist()
        except:
            # Fallback jika file tidak ada
            feature_names = ['Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)', 'Traffic Level', 
                           'Topping Density', 'Order Month', 'Pizza Complexity', 'Order Hour',
                           'Distance_Traffic', 'Complexity_Distance', 'Toppings_Density', 
                           'Is_Rush_Hour', 'Is_Weekend']
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please make sure you have trained your model and saved the following files:")
        st.info("- best_model.pkl")
        st.info("- scaler.pkl")
        st.info("- feature_names.csv (optional)")
        return None, None, None

# Helper function for feature engineering
def engineer_features(data):
    """Apply the same feature engineering as in training"""
    data_enhanced = data.copy()
    
    # Interaction features
    data_enhanced['Distance_Traffic'] = data['Distance (km)'] * data['Traffic Level']
    data_enhanced['Complexity_Distance'] = data['Pizza Complexity'] * data['Distance (km)']
    data_enhanced['Toppings_Density'] = data['Toppings Count'] * data['Topping Density']
    
    # Time-based features
    data_enhanced['Is_Rush_Hour'] = ((data['Order Hour'] >= 11) & (data['Order Hour'] <= 13)) | \
                                   ((data['Order Hour'] >= 18) & (data['Order Hour'] <= 20))
    data_enhanced['Is_Rush_Hour'] = data_enhanced['Is_Rush_Hour'].astype(int)
    
    # Weekend feature
    data_enhanced['Is_Weekend'] = ((data['Order Month'] == 6) | (data['Order Month'] == 7)).astype(int)
    
    return data_enhanced

# Load model at startup
model, scaler, feature_names = load_model_and_scaler()

# Sidebar for input
st.sidebar.header("🔧 Pizza Order Details")

# Pizza specifications
st.sidebar.subheader("Pizza Specifications")
pizza_size = st.sidebar.selectbox(
    "Pizza Size",
    options=[1, 2, 3, 4],
    format_func=lambda x: {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}[x],
    index=1
)

pizza_type = st.sidebar.selectbox(
    "Pizza Type",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    format_func=lambda x: {1: "Vegan", 2: "Non-Vegan", 3: "Cheese Burst", 4: "Gluten-Free", 5: "Stuffed Crust", 6: "Thin Crust",
                           7: "Deep Fish", 8: "Thai Chicken", 9: "Sicilian", 10: "BBQ Chicken", 11: "Margarita"}[x],
    index=0
)

toppings_count = st.sidebar.slider(
    "Number of Toppings",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="Total number of toppings on the pizza"
)

topping_density = st.sidebar.slider(
    "Topping Density",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="How densely packed the toppings are (0.1 = light, 2.0 = heavy)"
)

pizza_complexity = st.sidebar.slider(
    "Pizza Complexity",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
    help="Based on preparation difficulty"
)

# Delivery specifications
st.sidebar.subheader("Delivery Information")
distance = st.sidebar.slider(
    "Distance (km)",
    min_value=0.5,
    max_value=15.0,
    value=5.0,
    step=0.5,
    help="Distance from restaurant to delivery location"
)

traffic_level = st.sidebar.selectbox(
    "Traffic Level",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x],
    index=2
)

# Time specifications
st.sidebar.subheader("Order Timing")
order_hour = st.sidebar.slider(
    "Order Hour",
    min_value=0,
    max_value=23,
    value=19,
    step=1,
    format="%02d:00",
    help="Hour of the day when order is placed"
)

order_month = st.sidebar.selectbox(
    "Order Month",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    format_func=lambda x: {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}[x],
    index=4,
    help="Month of Order Placed"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Order Summary")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pizza Size': [pizza_size],
        'Pizza Type': [pizza_type],
        'Toppings Count': [toppings_count],
        'Distance (km)': [distance],
        'Traffic Level': [traffic_level],
        'Topping Density': [topping_density],
        'Order Month': [order_month],
        'Pizza Complexity': [pizza_complexity],
        'Order Hour': [order_hour]
    })
    
    # Engineer features
    input_enhanced = engineer_features(input_data)
    
    # Display order summary
    col1_1, col1_2, col1_3 = st.columns(3)
    
    with col1_1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🍕 Pizza Details</h4>
            <p><strong>Size:</strong> {["", "Small", "Medium", "Large", "Extra Large"][pizza_size]}</p>
            <p><strong>Type:</strong> {["", "Vegan", "Non-Vegan", "Cheese Burst", "Gluten-Free", "Stuffed Crust", "Thin Crust",
                                       "Deep Fish", "Thai Chicken", "Sicilian", "BBQ Chicken", "Margarita"][pizza_type]}</p>
            <p><strong>Toppings:</strong> {toppings_count}</p>
            <p><strong>Complexity:</strong> {pizza_complexity}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col1_2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🚚 Delivery Info</h4>
            <p><strong>Distance:</strong> {distance} km</p>
            <p><strong>Traffic:</strong> {["", "Low", "Medium", "High"][traffic_level]}</p>
            <p><strong>Rush Hour:</strong> {'Yes' if input_enhanced['Is_Rush_Hour'].iloc[0] else 'No'}</p>
            <p><strong>Weekend:</strong> {'Yes' if input_enhanced['Is_Weekend'].iloc[0] else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col1_3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏰ Timing</h4>
            <p><strong>Order Hour:</strong> {order_hour:02d}:00</p>
            <p><strong>Month:</strong> {["", "January", "February", "March", "April", "May", "June", "July", "August",
                                        "September", "October", "November", "December"][order_month]}</p>
            <p><strong>Topping Density:</strong> {topping_density}</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.header("🔮 Prediction")
    
    # Show model status
    if model is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model not found!")
    
    # Predict button
    if st.button("Predict Delivery Time", type="primary", use_container_width=True):
        
        if model is not None and scaler is not None:
            try:
                # Scale the input data
                input_scaled = scaler.transform(input_enhanced)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Estimated Delivery Time</h3>
                    <h1>{prediction:.1f} minutes</h1>
                    <p>≈ {int(prediction//60)}h {int(prediction%60)}m</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicators
                if prediction <= 30:
                    st.success("🟢 Fast delivery expected!")
                elif prediction <= 45:
                    st.warning("🟡 Moderate delivery time")
                else:
                    st.error("🔴 Longer delivery time expected")
                
                # Show input features used
                with st.expander("📋 Features Used for Prediction"):
                    st.dataframe(input_enhanced.T, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check if your model and scaler are compatible with the input data.")
                
        else:
            # Demo prediction (fallback)
            demo_prediction = 25 + (distance * 2) + (traffic_level * 3) + (toppings_count * 0.5) + (pizza_complexity * 2)
            if input_enhanced['Is_Rush_Hour'].iloc[0]:
                demo_prediction += 5
            if input_enhanced['Is_Weekend'].iloc[0]:
                demo_prediction += 3
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Demo Prediction</h3>
                <h1>{demo_prediction:.1f} minutes</h1>
                <p>≈ {int(demo_prediction//60)}h {int(demo_prediction%60)}m</p>
                <small>⚠️ This is a demo prediction. Please train and load your model!</small>
            </div>
            """, unsafe_allow_html=True)

# Feature importance visualization
st.header("📈 Feature Analysis")

if model is not None and hasattr(model, 'feature_importances_'):
    # Use actual feature importance from loaded model
    if feature_names:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Feature Importance (Actual Model)",
            labels={'importance': 'Importance', 'feature': 'Features'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature names not available. Please ensure feature_names.csv is in the same directory.")
else:
    # Fallback to demo feature importance
    features = ['Distance (km)', 'Traffic Level', 'Pizza Complexity', 'Toppings Count', 
               'Is_Rush_Hour', 'Distance_Traffic', 'Order Hour', 'Pizza Size', 
               'Topping Density', 'Is_Weekend']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01]

    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance (Demo)",
        labels={'x': 'Importance', 'y': 'Features'},
        color=importance,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tips section
st.header("💡 Tips for Faster Delivery")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🚀 Speed Up Your Order
    - **Order during off-peak hours** (avoid 12-2 PM and 6-8 PM)
    - **Choose simpler pizzas** with fewer toppings
    - **Order on weekdays** for potentially faster service
    - **Consider smaller sizes** for quicker preparation
    """)

with col2:
    st.markdown("""
    ### 📍 Location Matters
    - **Closer locations** = faster delivery
    - **Check traffic conditions** before ordering
    - **Provide clear address** to avoid delays
    - **Consider pickup** during heavy traffic
    """)

# Footer
st.markdown("---")
st.markdown("Made by Group 9 | Pizza Delivery Time Predictor v1.0")
