import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="🍕 Pizza Delivery Time Predictor",
    page_icon="🍕",
    layout="wide"
)

# Load model function
@st.cache_resource
def load_model():
    """Load the best available model"""
    model_candidates = [
        'pizza_delivery_CORRECT_TARGET.pkl',
        'pizza_delivery_model_FINAL.pkl', 
        'realistic_rf_model_FIXED.pkl',
        'realistic_rf_model.pkl'
    ]
    
    for model_file in model_candidates:
        if not os.path.exists(model_file):
            continue
            
        try:
            with open(model_file, 'rb') as f:
                model_info = pickle.load(f)
            
            if not isinstance(model_info, dict):
                continue
                
            required_keys = ['model', 'features', 'model_performance']
            if not all(key in model_info for key in required_keys):
                continue
            
            expected_features = [
                'Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                'Distance (km)', 'Topping Density', 'Traffic Level', 
                'Is Peak Hour', 'Is Weekend'
            ]
            
            if model_info['features'] != expected_features:
                continue
            
            file_age = (datetime.now().timestamp() - os.path.getmtime(model_file)) / 3600
            
            return {
                'model': model_info['model'],
                'features': model_info['features'],
                'performance': model_info['model_performance'],
                'target_column': model_info.get('target_column', 'Unknown'),
                'correct_target_used': model_info.get('correct_target_used', None),
                'model_type': model_info.get('model_type', 'Unknown'),
                'noise_added': model_info.get('noise_added', False),
                'prediction_std': model_info.get('prediction_std', 0),
                'filename': model_file,
                'age_hours': file_age
            }, None
            
        except Exception as e:
            continue
    
    return None, "❌ No valid model found. Please run training script first."

def main():
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("---")
    
    # Load model
    model_data, error_message = load_model()
    
    if error_message:
        st.error(error_message)
        st.info("📝 **Steps to fix:**")
        st.code("python QUICK_FIX_training.py")
        st.code("streamlit run app_FIXED.py")
        return
    
    # Model status
    st.success("✅ Model loaded successfully!")
    
    target_col = model_data['target_column']
    correct_target = model_data['correct_target_used']
    
    if correct_target is True:
        st.success(f"✅ **Correct Target**: '{target_col}' (Actual delivery duration)")
    elif correct_target is False:
        st.error(f"🚨 **Wrong Target**: '{target_col}' (Estimated duration)")
    else:
        st.warning(f"⚠️ **Target**: '{target_col}' (Not validated)")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        perf = model_data['performance']
        st.metric("Model File", model_data['filename'])
        st.metric("Accuracy (R²)", f"{perf.get('test_r2', 0):.1%}")
        st.metric("Average Error", f"±{perf.get('test_mae', 0):.1f} min")
        
        if perf.get('test_r2', 0) > 0.7:
            st.success("✅ Good performance")
        else:
            st.warning("📊 Moderate performance")
    
    # Main interface
    st.subheader("🔧 Make a Prediction")
    
    # Form with proper structure
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🍕 Pizza Details")
            
            pizza_complexity = st.selectbox(
                "Pizza Complexity",
                options=[1, 2, 3, 4, 5],
                index=2,  # Default to 3
                format_func=lambda x: f"{x} - {['', 'Simple', 'Easy', 'Medium', 'Complex', 'Very Complex'][x]}"
            )
            
            topping_density = st.selectbox(
                "Topping Density", 
                options=[1, 2, 3, 4, 5],
                index=1,  # Default to 2
                format_func=lambda x: f"{x} - {['', 'Light', 'Medium-Light', 'Medium', 'Heavy', 'Very Heavy'][x]}"
            )
            
            restaurant_avg_time = st.number_input(
                "Restaurant Avg Time (minutes)",
                min_value=10,
                max_value=60,
                value=25,
                step=5
            )
        
        with col2:
            st.markdown("#### 🚚 Delivery Details")
            
            distance = st.number_input(
                "Distance (km)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5
            )
            
            traffic_level = st.selectbox(
                "Traffic Level",
                options=[1, 2, 3, 4, 5],
                index=1,  # Default to 2
                format_func=lambda x: f"{x} - {['', 'Clear', 'Light', 'Medium', 'Heavy', 'Very Heavy'][x]}"
            )
            
            order_hour = st.number_input(
                "Order Hour (0-23)",
                min_value=0,
                max_value=23,
                value=18,
                step=1
            )
        
        # Auto-calculate peak hour and weekend
        is_peak_hour = 1 if (11 <= order_hour <= 14) or (17 <= order_hour <= 20) else 0
        is_weekend = 0  # Default to weekday
        
        # Show auto-calculated values
        st.markdown("#### ⏰ Auto-Calculated")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Peak Hour: {'✅ Yes' if is_peak_hour else '❌ No'}")
        with col2:
            st.info(f"Weekend: {'✅ Yes' if is_weekend else '❌ No'}")
        
        # Manual override options
        st.markdown("#### 🔧 Manual Overrides (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            override_peak = st.checkbox("Override Peak Hour Detection")
            if override_peak:
                is_peak_hour = st.selectbox(
                    "Manual Peak Hour",
                    options=[0, 1],
                    index=is_peak_hour,
                    format_func=lambda x: "Yes (Busy)" if x == 1 else "No (Quiet)"
                )
        
        with col2:
            override_weekend = st.checkbox("Override Weekend Detection")  
            if override_weekend:
                is_weekend = st.selectbox(
                    "Manual Weekend",
                    options=[0, 1], 
                    index=is_weekend,
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )
        
        # Submit button - REQUIRED for forms
        submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary")
        
        # Process prediction when form is submitted
        if submitted:
            try:
                # Prepare input data
                input_data = np.array([[
                    pizza_complexity,
                    order_hour,
                    restaurant_avg_time,
                    distance,
                    topping_density,
                    traffic_level,
                    is_peak_hour,
                    is_weekend
                ]])
                
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Main result
                if correct_target is True:
                    st.success(f"### 🕐 **Predicted Delivery Time: {predicted_duration:.1f} minutes**")
                else:
                    st.warning(f"### ⚠️ **Predicted Time: {predicted_duration:.1f} minutes**")
                    if correct_target is False:
                        st.caption("(May be based on estimated duration)")
                
                # Time conversion
                hours = int(predicted_duration // 60)
                minutes = int(predicted_duration % 60)
                if hours > 0:
                    st.info(f"📅 **Total Time**: {hours} hour(s) {minutes} minute(s)")
                else:
                    st.info(f"📅 **Total Time**: {minutes} minute(s)")
                
                # Confidence range
                uncertainty = model_data.get('prediction_std', model_data['performance'].get('test_mae', 3))
                if uncertainty > 0:
                    lower = max(5, predicted_duration - uncertainty)
                    upper = predicted_duration + uncertainty
                    st.info(f"📊 **Confidence Range**: {lower:.1f} - {upper:.1f} minutes")
                
                # Time category
                if predicted_duration <= 20:
                    st.success("🟢 **Fast Delivery** - Excellent!")
                elif predicted_duration <= 35:
                    st.warning("🟡 **Normal Delivery** - Standard time")
                else:
                    st.error("🔴 **Slow Delivery** - Longer than usual")
                
                # Detailed breakdown
                with st.expander("🔍 Detailed Analysis"):
                    # Input summary
                    input_df = pd.DataFrame({
                        'Feature': model_data['features'],
                        'Value': input_data[0],
                        'Description': [
                            f"Complexity {pizza_complexity}/5",
                            f"{order_hour:02d}:00 ({'Peak' if is_peak_hour else 'Off-peak'})",
                            f"{restaurant_avg_time} min prep",
                            f"{distance} km distance", 
                            f"Density {topping_density}/5",
                            f"Traffic {traffic_level}/5",
                            "Peak hour" if is_peak_hour else "Off-peak",
                            "Weekend" if is_weekend else "Weekday"
                        ]
                    })
                    st.dataframe(input_df, use_container_width=True)
                    
                    # Model info
                    st.write("**Model Performance:**")
                    perf = model_data['performance']
                    st.write(f"- Target: {target_col}")
                    st.write(f"- Accuracy: {perf.get('test_r2', 0):.1%}")
                    st.write(f"- Error: ±{perf.get('test_mae', 0):.1f} min")
                    st.write(f"- Model: {model_data['model_type']}")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.write("**Debug info:**")
                st.write(f"- Input shape: {input_data.shape if 'input_data' in locals() else 'Not created'}")
                st.write(f"- Model type: {type(model_data['model'])}")
    
    # Testing section
    st.markdown("---")
    st.subheader("🧪 Model Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test with Sample Data"):
            test_data = np.array([[3, 14, 25, 5, 2, 3, 1, 0]])
            test_pred = model_data['model'].predict(test_data)[0]
            st.info(f"**Test Result**: {test_pred:.1f} minutes")
            st.caption("Sample: Complexity=3, Hour=14, RestTime=25, Distance=5, Density=2, Traffic=3, Peak=1, Weekend=0")
    
    with col2:
        if st.button("📊 Show Model Details"):
            st.json({
                "filename": model_data['filename'],
                "target": target_col,
                "correct_target": correct_target,
                "r2_score": f"{model_data['performance'].get('test_r2', 0):.3f}",
                "mae_minutes": f"{model_data['performance'].get('test_mae', 0):.1f}",
                "features": len(model_data['features'])
            })
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This Predictor")
    
    st.markdown(f"""
    **🎯 Purpose**: Predicts actual pizza delivery time based on order characteristics.
    
    **📊 Current Model**: 
    - File: `{model_data['filename']}`
    - Target: `{target_col}`
    - Accuracy: {model_data['performance'].get('test_r2', 0):.1%}
    - Average Error: ±{model_data['performance'].get('test_mae', 0):.1f} minutes
    
    **🔧 Features Used**:
    - Pizza complexity and topping density
    - Restaurant preparation patterns
    - Distance and traffic conditions  
    - Time-based factors (peak hours, weekends)
    
    **✅ Quality**: {'Target validated as actual delivery duration' if correct_target is True else 'Target validation needed' if correct_target is None else 'Using estimated duration (not ideal)'}
    """)

if __name__ == "__main__":
    main()
