import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 QUICK FIX TRAINING SCRIPT - CORRECT COLUMN NAME")
print("=" * 70)

# Load data
try:
    data = pd.read_excel('Train Data.xlsx')
    print(f"✅ Data loaded: {data.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Show all columns
print(f"\n📊 ALL COLUMNS:")
for i, col in enumerate(data.columns):
    print(f"{i+1:2d}. '{col}'")

# Find the correct target column
delivery_candidates = [
    'Delivery Duration (min)',  # lowercase min (from debug output)
    'Delivery Duration (Min)',  # uppercase Min
    'Delivery Duration'         # without parentheses
]

estimated_candidates = [
    'Estimated Duration (Min)',
    'Estimated Duration (min)', 
    'Estimated Duration'
]

# Find correct delivery column
delivery_col = None
for candidate in delivery_candidates:
    if candidate in data.columns:
        delivery_col = candidate
        break

# Find estimated column (to avoid using it)
estimated_col = None
for candidate in estimated_candidates:
    if candidate in data.columns:
        estimated_col = candidate
        break

print(f"\n🎯 TARGET COLUMN DETECTION:")
print("-" * 30)
if delivery_col:
    print(f"✅ FOUND DELIVERY COLUMN: '{delivery_col}'")
else:
    print("❌ No delivery column found!")
    available_duration = [col for col in data.columns if 'duration' in col.lower()]
    print(f"Available duration columns: {available_duration}")
    exit()

if estimated_col:
    print(f"⚠️  FOUND ESTIMATED COLUMN: '{estimated_col}' (will NOT use this)")

target = delivery_col
print(f"\n✅ USING TARGET: '{target}'")

# Compare if both exist
if estimated_col and delivery_col:
    print(f"\n📊 COMPARISON:")
    print(f"Delivery mean: {data[delivery_col].mean():.1f} min")
    print(f"Estimated mean: {data[estimated_col].mean():.1f} min")
    print(f"Difference: {data[delivery_col].mean() - data[estimated_col].mean():+.1f} min")
    
    # Check if identical (suspicious)
    if data[delivery_col].equals(data[estimated_col]):
        print("🚨 WARNING: Delivery and Estimated are identical!")
    else:
        diff_count = (data[delivery_col] != data[estimated_col]).sum()
        print(f"Different values: {diff_count}/{len(data)} ({diff_count/len(data)*100:.1f}%)")

# Add engineered features
data['Is Peak Hour'] = np.where(((data['Order Hour'] >= 11) & (data['Order Hour'] <= 14)) |
                                 ((data['Order Hour'] >= 17) & (data['Order Hour'] <= 20)), 1, 0)
data['Is Weekend'] = np.where(data['Order Month'].isin([6, 7, 8, 9]), 1, 0)

# Define features
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Distance (km)', 'Topping Density', 'Traffic Level', 
            'Is Peak Hour', 'Is Weekend']

print(f"\n📊 TARGET ANALYSIS:")
print(f"Target: '{target}'")
print(f"Min: {data[target].min():.1f}, Max: {data[target].max():.1f}")
print(f"Mean: {data[target].mean():.1f}, Std: {data[target].std():.1f}")

# Prepare data
X = data[features].copy()
y = data[target].copy()

# Clean data
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]

print(f"\nCleaned data: {X_clean.shape}")

# Check determinism
feature_target_df = X_clean.copy()
feature_target_df['target'] = y_clean
grouped = feature_target_df.groupby(features)['target'].nunique()
perfect_matches = (grouped == 1).sum()
total_groups = len(grouped)
determinism_ratio = perfect_matches / total_groups

print(f"Determinism: {determinism_ratio*100:.1f}%")

# Add noise if needed
if determinism_ratio > 0.8:
    print("Adding realistic noise...")
    noise_std = y_clean.std() * 0.05
    y_with_noise = y_clean + np.random.normal(0, noise_std, size=len(y_clean))
    y_with_noise = np.maximum(y_with_noise, 5)
    y_final = y_with_noise
    use_noise = True
else:
    y_final = y_clean
    use_noise = False

# Split data
y_bins = pd.cut(y_final, bins=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_final, test_size=0.2, random_state=42, stratify=y_bins
)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# Train model
print(f"\n🤖 TRAINING MODEL:")
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

# Train and evaluate
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 FEATURE IMPORTANCE:")
for _, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")

# Calculate prediction std
residuals = y_test - y_pred_test
residual_std = residuals.std()

# Save model with correct metadata
model_info = {
    'model': model,
    'features': features,
    'feature_names': features,
    'n_features': len(features),
    'target_column': target,  # This will be the correct column name
    'correct_target_used': True,  # We explicitly verified this
    'model_performance': {
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'train_mae': train_mae
    },
    'feature_importance': feature_importance.to_dict(),
    'model_type': 'RandomForest',
    'noise_added': use_noise,
    'prediction_std': residual_std,
    'training_info': {
        'data_shape': data.shape,
        'clean_data_shape': X_clean.shape,
        'deterministic_ratio': determinism_ratio,
        'target_stats': {
            'mean': float(y_clean.mean()),
            'std': float(y_clean.std()),
            'min': float(y_clean.min()),
            'max': float(y_clean.max())
        }
    },
    'validation_info': {
        'delivery_column_used': delivery_col,
        'estimated_column_found': estimated_col,
        'targets_compared': estimated_col is not None,
        'estimated_vs_actual_diff': float(data[target].mean() - data[estimated_col].mean()) if estimated_col else None
    }
}

# Save model
model_filename = 'pizza_delivery_CORRECT_TARGET.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_info, f)

print(f"\n💾 MODEL SAVED: '{model_filename}'")
print(f"📊 Target: '{target}' (actual delivery duration)")
print(f"✅ Performance: R²={test_r2:.3f}, MAE={test_mae:.1f}min")

# Test prediction
test_input = np.array([[3, 14, 25, 5, 2, 3, 1, 0]])
test_prediction = model.predict(test_input)[0]
print(f"\n🧪 TEST PREDICTION: {test_prediction:.1f} minutes")

print(f"\n✅ TRAINING COMPLETE!")
print("🚀 Now run: streamlit run 4_app_final.py")
print(f"📁 Model file: {model_filename}")