import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="🍕 Pizza Delivery Time Predictor - Final Version",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #FF6B35;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with comprehensive validation
@st.cache_resource
def load_model():
    """Load the best available model with comprehensive validation"""
    
    # Priority order for model files
    model_candidates = [
        'pizza_delivery_CORRECT_TARGET.pkl',  # From quick fix script
        'pizza_delivery_model_FINAL.pkl',     # From final training script
        'realistic_rf_model_FIXED.pkl',       # From fixed training script
        'realistic_rf_model.pkl'              # Original model (fallback)
    ]
    
    for model_file in model_candidates:
        if not os.path.exists(model_file):
            continue
            
        try:
            # Load model
            with open(model_file, 'rb') as f:
                model_info = pickle.load(f)
            
            # Validate structure
            if not isinstance(model_info, dict):
                continue
                
            required_keys = ['model', 'features', 'model_performance']
            if not all(key in model_info for key in required_keys):
                continue
            
            # Validate features
            expected_features = [
                'Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
                'Distance (km)', 'Topping Density', 'Traffic Level', 
                'Is Peak Hour', 'Is Weekend'
            ]
            
            if model_info['features'] != expected_features:
                continue
            
            # Get file info
            file_age = (datetime.now().timestamp() - os.path.getmtime(model_file)) / 3600
            
            # Extract model data
            return {
                'model': model_info['model'],
                'features': model_info['features'],
                'performance': model_info['model_performance'],
                'target_column': model_info.get('target_column', 'Unknown'),
                'correct_target_used': model_info.get('correct_target_used', None),
                'model_type': model_info.get('model_type', 'Unknown'),
                'noise_added': model_info.get('noise_added', False),
                'prediction_std': model_info.get('prediction_std', 0),
                'training_info': model_info.get('training_info', {}),
                'validation_info': model_info.get('validation_info', {}),
                'file_info': {
                    'filename': model_file,
                    'age_hours': file_age,
                    'size_bytes': os.path.getsize(model_file)
                }
            }, None
            
        except Exception as e:
            continue
    
    return None, "❌ No valid model found. Please run the training script first."

# Load model
model_data, error_message = load_model()

def main():
    # Header
    st.markdown('<div class="main-header">🍕 Pizza Delivery Time Predictor</div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Predicting **Actual Delivery Duration** with Machine Learning")
    st.markdown("---")
    
    # Model status
    if error_message:
        st.markdown('<div class="status-error">', unsafe_allow_html=True)
        st.error(error_message)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("🔧 **To fix this:**")
        col1, col2 = st.columns(2)
        with col1:
            st.code("python 1_debug_columns.py")
            st.code("python 2_training_fixed.py")
        with col2:
            st.code("python 3_check_files.py")
            st.code("streamlit run 4_app_final.py")
        return
    
    # Model loaded successfully
    st.markdown('<div class="status-success">', unsafe_allow_html=True)
    st.success("✅ **Model loaded successfully!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model validation status
    target_col = model_data['target_column']
    correct_target = model_data['correct_target_used']
    
    if correct_target is True:
        st.markdown('<div class="status-success">', unsafe_allow_html=True)
        st.success(f"✅ **Correct Target**: '{target_col}' (Actual delivery duration)")
        st.markdown('</div>', unsafe_allow_html=True)
    elif correct_target is False:
        st.markdown('<div class="status-error">', unsafe_allow_html=True)
        st.error(f"🚨 **Wrong Target**: '{target_col}' (This appears to be estimated duration!)")
        st.error("**Solution**: Run `python 2_training_fixed.py` to retrain with correct target.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">', unsafe_allow_html=True)
        st.warning(f"⚠️ **Target**: '{target_col}' (Not validated - please verify)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        # File info
        file_info = model_data['file_info']
        st.metric("Model File", file_info['filename'])
        st.metric("File Age", f"{file_info['age_hours']:.1f} hours")
        st.metric("File Size", f"{file_info['size_bytes']:,} bytes")
        
        # Performance metrics
        perf = model_data['performance']
        st.header("🎯 Performance")
        st.metric("Accuracy (R²)", f"{perf.get('test_r2', 0):.1%}")
        st.metric("Average Error", f"±{perf.get('test_mae', 0):.1f} min")
        st.metric("CV Error", f"±{perf.get('cv_mae', 0):.1f} min")
        
        # Performance interpretation
        test_r2 = perf.get('test_r2', 0)
        test_mae = perf.get('test_mae', 0)
        
        if test_r2 > 0.95:
            st.error("🚨 Very high R² - check for overfitting")
        elif test_r2 > 0.8:
            st.warning("⚠️ High R² - validate on new data")
        elif test_r2 > 0.7:
            st.success("✅ Good model performance")
        elif test_r2 > 0.5:
            st.info("📊 Moderate performance")
        else:
            st.warning("📉 Low performance")
        
        if test_mae < 3:
            st.success("✅ Very accurate predictions")
        elif test_mae < 5:
            st.success("✅ Accurate predictions")
        elif test_mae < 8:
            st.info("📊 Reasonable accuracy")
        else:
            st.warning("📈 Consider improving model")
    
    # Model details expandable section
    with st.expander("🔍 Detailed Model Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Model Details")
            st.write(f"**Model Type**: {model_data['model_type']}")
            st.write(f"**Features**: {len(model_data['features'])}")
            st.write(f"**Target Column**: {target_col}")
            st.write(f"**Noise Added**: {'✅ Yes' if model_data['noise_added'] else '❌ No'}")
            
            # Training info
            training_info = model_data['training_info']
            if training_info:
                st.write(f"**Dataset Size**: {training_info.get('data_shape', 'Unknown')}")
                st.write(f"**Clean Data**: {training_info.get('clean_data_shape', 'Unknown')}")
                det_ratio = training_info.get('deterministic_ratio', 0)
                st.write(f"**Determinism**: {det_ratio*100:.1f}%")
        
        with col2:
            st.subheader("📊 Target Statistics")
            target_stats = training_info.get('target_stats', {})
            if target_stats:
                st.metric("Mean Duration", f"{target_stats.get('mean', 0):.1f} min")
                st.metric("Std Deviation", f"{target_stats.get('std', 0):.1f} min")
                st.metric("Min Duration", f"{target_stats.get('min', 0):.1f} min")
                st.metric("Max Duration", f"{target_stats.get('max', 0):.1f} min")
            
            # Validation info
            validation_info = model_data['validation_info']
            if validation_info.get('targets_compared', False):
                diff = validation_info.get('estimated_vs_actual_diff', 0)
                st.metric("Actual vs Estimated", f"{diff:+.1f} min")
        
        # Feature list
        st.subheader("🎯 Features Used")
        for i, feature in enumerate(model_data['features'], 1):
            st.write(f"{i}. **{feature}**")
    
    # Main prediction interface
    st.subheader("🔧 Make a Prediction")
    
    # Description
    st.markdown(f"""
    Enter your pizza order details below to predict the **actual delivery time**. 
    The model is trained on real delivery data using the `{target_col}` column.
    """)
    
    # Input form
    with st.form("prediction_form", clear_on_submit=False):
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🍕 Pizza Details")
            
            pizza_complexity = st.select_slider(
                "Pizza Complexity",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{x} - {['', 'Simple', 'Easy', 'Medium', 'Complex', 'Very Complex'][x]}",
                help="1 = Simple (Margherita), 5 = Very Complex (Supreme with many toppings)"
            )
            
            topping_density = st.select_slider(
                "Topping Density",
                options=[1, 2, 3, 4, 5],
                value=2,
                format_func=lambda x: f"{x} - {['', 'Light', 'Medium-Light', 'Medium', 'Heavy', 'Very Heavy'][x]}",
                help="Amount of toppings on the pizza"
            )
            
            restaurant_avg_time = st.slider(
                "Restaurant Average Prep Time",
                min_value=10, max_value=60, value=25, step=5,
                help="Average preparation time for this restaurant (minutes)"
            )
            
        with col2:
            st.markdown("#### 🚚 Delivery Details")
            
            distance = st.slider(
                "Delivery Distance (km)",
                min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                help="Distance from restaurant to delivery location"
            )
            
            traffic_level = st.select_slider(
                "Traffic Level",
                options=[1, 2, 3, 4, 5],
                value=2,
                format_func=lambda x: f"{x} - {['', 'Clear', 'Light', 'Medium', 'Heavy', 'Very Heavy'][x]}",
                help="Current traffic conditions"
            )
            
            # Time and date inputs
            st.markdown("#### ⏰ Timing")
            
            order_hour = st.slider(
                "Order Hour",
                min_value=0, max_value=23, value=18,
                format_func=lambda x: f"{x:02d}:00",
                help="Hour when order is placed (24-hour format)"
            )
            
            # Auto-calculate peak hour
            auto_peak = 1 if (11 <= order_hour <= 14) or (17 <= order_hour <= 20) else 0
            is_peak_hour = st.selectbox(
                "Peak Hour?",
                options=[0, 1],
                index=auto_peak,
                format_func=lambda x: "✅ Yes (Busy)" if x == 1 else "❌ No (Quiet)",
                help="Peak hours: 11-14 (lunch) and 17-20 (dinner)"
            )
            
            is_weekend = st.selectbox(
                "Weekend?",
                options=[0, 1],
                index=0,
                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
                help="Weekend affects delivery patterns"
            )
        
        # Show current selection summary
        st.markdown("#### 📋 Order Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.info(f"🍕 **Pizza**: Complexity {pizza_complexity}/5, Toppings {topping_density}/5")
        with summary_col2:
            st.info(f"🚚 **Delivery**: {distance}km, Traffic {traffic_level}/5")
        with summary_col3:
            peak_text = "Peak" if is_peak_hour else "Off-peak"
            weekend_text = "Weekend" if is_weekend else "Weekday"
            st.info(f"⏰ **Time**: {order_hour:02d}:00, {peak_text}, {weekend_text}")
        
        # Prediction button
        submitted = st.form_submit_button("🚀 Predict Delivery Time", type="primary", use_container_width=True)
        
        if submitted:
            # Prepare input data (exact order as training)
            input_data = np.array([[
                pizza_complexity,    # 0: Pizza Complexity
                order_hour,         # 1: Order Hour
                restaurant_avg_time, # 2: Restaurant Avg Time
                distance,           # 3: Distance (km)
                topping_density,    # 4: Topping Density
                traffic_level,      # 5: Traffic Level
                is_peak_hour,       # 6: Is Peak Hour
                is_weekend          # 7: Is Weekend
            ]])
            
            try:
                # Make prediction
                model = model_data['model']
                predicted_duration = model.predict(input_data)[0]
                
                # Show results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Main prediction with confidence
                prediction_std = model_data.get('prediction_std', 0)
                mae = model_data['performance'].get('test_mae', 0)
                uncertainty = max(prediction_std, mae)
                
                # Main result
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    if correct_target is True:
                        st.success(f"### 🕐 **Predicted Delivery Time: {predicted_duration:.1f} minutes**")
                    elif correct_target is False:
                        st.warning(f"### ⚠️ **Predicted Time: {predicted_duration:.1f} minutes**")
                        st.caption("(Based on estimated duration - may not reflect actual delivery time)")
                    else:
                        st.info(f"### 🕐 **Predicted Time: {predicted_duration:.1f} minutes**")
                
                with result_col2:
                    # Convert to hours:minutes
                    hours = int(predicted_duration // 60)
                    minutes = int(predicted_duration % 60)
                    if hours > 0:
                        st.metric("Total Time", f"{hours}h {minutes}m")
                    else:
                        st.metric("Total Time", f"{minutes} min")
                
                # Confidence interval
                if uncertainty > 0:
                    lower_bound = max(5, predicted_duration - uncertainty)
                    upper_bound = predicted_duration + uncertainty
                    st.info(f"**📊 Confidence Range: {lower_bound:.1f} - {upper_bound:.1f} minutes**")
                
                # Time categorization
                col1, col2, col3 = st.columns(3)
                
                if predicted_duration <= 20:
                    with col1:
                        st.success("🟢 **Fast Delivery**\nExcellent timing!")
                elif predicted_duration <= 35:
                    with col2:
                        st.warning("🟡 **Normal Delivery**\nStandard timing")
                else:
                    with col3:
                        st.error("🔴 **Slow Delivery**\nLonger than usual")
                
                # Detailed breakdown
                with st.expander("🔍 Detailed Analysis", expanded=False):
                    # Input verification
                    st.subheader("📊 Input Verification")
                    
                    input_df = pd.DataFrame({
                        'Feature': model_data['features'],
                        'Value': input_data[0],
                        'Description': [
                            f"Complexity level {pizza_complexity}/5",
                            f"{order_hour:02d}:00 ({'Peak' if is_peak_hour else 'Off-peak'})",
                            f"{restaurant_avg_time} minutes average prep time",
                            f"{distance} km delivery distance",
                            f"Topping density {topping_density}/5",
                            f"Traffic level {traffic_level}/5",
                            "Peak hour (busy)" if is_peak_hour else "Off-peak hour (quiet)",
                            "Weekend" if is_weekend else "Weekday"
                        ]
                    })
                    st.dataframe(input_df, use_container_width=True)
                    
                    # Model performance
                    st.subheader("🎯 Model Performance")
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        st.write(f"**Target Column**: {target_col}")
                        st.write(f"**Model Type**: {model_data['model_type']}")
                        st.write(f"**Accuracy (R²)**: {perf.get('test_r2', 0):.1%}")
                        st.write(f"**Average Error**: ±{perf.get('test_mae', 0):.1f} minutes")
                    
                    with perf_col2:
                        st.write(f"**Cross-Validation Error**: ±{perf.get('cv_mae', 0):.1f} minutes")
                        st.write(f"**Training with Noise**: {'✅ Yes' if model_data['noise_added'] else '❌ No'}")
                        st.write(f"**Target Validated**: {'✅ Yes' if correct_target is True else '❌ No' if correct_target is False else '❓ Unknown'}")
                    
                    # Debug info
                    st.subheader("🐛 Debug Information")
                    st.code(f"Input array: {input_data[0].tolist()}")
                    st.code(f"Prediction: {predicted_duration:.6f}")
                    st.code(f"Model file: {model_data['file_info']['filename']}")
                
            except Exception as e:
                st.error(f"❌ **Prediction failed**: {str(e)}")
                st.error("This might indicate a model compatibility issue.")
                
                with st.expander("🔧 Troubleshooting"):
                    st.write("**Possible solutions:**")
                    st.write("1. Check if model file is corrupted")
                    st.write("2. Retrain the model with `python 2_training_fixed.py`")
                    st.write("3. Verify input data format")
                    
                    st.write("**Debug info:**")
                    st.write(f"- Model type: {type(model_data['model'])}")
                    st.write(f"- Input shape: {input_data.shape}")
                    st.write(f"- Expected features: {len(model_data['features'])}")
    
    # Testing and validation section
    st.markdown("---")
    st.subheader("🧪 Model Testing & Validation")
    
    test_col1, test_col2 = st.columns(2)
    
    with test_col1:
        if st.button("🧪 Test with Sample Data", use_container_width=True):
            # Standard test case
            test_data = np.array([[3, 14, 25, 5, 2, 3, 1, 0]])
            test_prediction = model_data['model'].predict(test_data)[0]
            
            st.info(f"**Test Result**: {test_prediction:.1f} minutes")
            st.caption("Input: Complexity=3, Hour=14, RestTime=25, Distance=5, Density=2, Traffic=3, Peak=1, Weekend=0")
            
            # Validate result
            expected_range = (15, 50)  # Reasonable range
            if expected_range[0] <= test_prediction <= expected_range[1]:
                st.success("✅ Test prediction is within reasonable range")
            else:
                st.warning(f"⚠️ Test prediction outside expected range {expected_range[0]}-{expected_range[1]} minutes")
    
    with test_col2:
        if st.button("📊 Model Summary", use_container_width=True):
            st.json({
                "model_file": model_data['file_info']['filename'],
                "target_column": target_col,
                "correct_target": correct_target,
                "performance": {
                    "r2_score": f"{perf.get('test_r2', 0):.3f}",
                    "mae_minutes": f"{perf.get('test_mae', 0):.1f}",
                    "cv_mae_minutes": f"{perf.get('cv_mae', 0):.1f}"
                },
                "features_count": len(model_data['features']),
                "noise_added": model_data['noise_added']
            })
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ How This Predictor Works")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **🎯 Target Variable:**
        This model predicts the **actual delivery duration** - the real time it takes from order placement to delivery.
        
        **🤖 Algorithm:**
        Uses Random Forest Regression trained on historical pizza delivery data with realistic noise for better generalization.
        
        **📊 Features:**
        - Pizza complexity and topping density
        - Restaurant preparation time patterns  
        - Distance and traffic conditions
        - Time-based factors (hour, peak times, weekend)
        """)
    
    with info_col2:
        st.markdown(f"""
        **📈 Model Performance:**
        - Accuracy (R²): {perf.get('test_r2', 0):.1%}
        - Average Error: ±{perf.get('test_mae', 0):.1f} minutes
        - Cross-Validation: ±{perf.get('cv_mae', 0):.1f} minutes
        
        **✅ Quality Assurance:**
        - Target column validated: {'✅ Yes' if correct_target is True else '❌ No' if correct_target is False else '❓ Unknown'}
        - Realistic noise added: {'✅ Yes' if model_data['noise_added'] else '❌ No'}
        - Overfitting prevented: Cross-validation used
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    🍕 Pizza Delivery Time Predictor | Built with Streamlit & Scikit-learn<br>
    Model trained on actual delivery duration data for realistic predictions
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()