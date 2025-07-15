import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import streamlit as st

# Load data
data = pd.read_excel('Train Data.xlsx')

# Pilih fitur yang relevan
features = ['Pizza Size', 'Pizza Type', 'Toppings Count', 'Distance (km)', 'Traffic Level', 
            'Topping Density', 'Order Month', 'Pizza Complexity', 'Order Hour']

# Ambil data fitur relevan dan target 'Delivery Duration (min)'
X = data[features].dropna()
y = data['Delivery Duration (min)']

# 1. FEATURE ENGINEERING - Tambah fitur baru
# Interaction features
X_enhanced = X.copy()
X_enhanced['Distance_Traffic'] = X['Distance (km)'] * X['Traffic Level']
X_enhanced['Complexity_Distance'] = X['Pizza Complexity'] * X['Distance (km)']
X_enhanced['Toppings_Density'] = X['Toppings Count'] * X['Topping Density']

# Time-based features
X_enhanced['Is_Rush_Hour'] = ((X['Order Hour'] >= 11) & (X['Order Hour'] <= 13)) | \
                             ((X['Order Hour'] >= 18) & (X['Order Hour'] <= 20))
X_enhanced['Is_Rush_Hour'] = X_enhanced['Is_Rush_Hour'].astype(int)

# Weekend feature (assuming Order Month is actually day of week)
X_enhanced['Is_Weekend'] = ((X['Order Month'] == 6) | (X['Order Month'] == 7)).astype(int)

# 2. OUTLIER DETECTION DAN REMOVAL
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
outlier_mask = (y >= lower_bound) & (y <= upper_bound)
X_clean = X_enhanced[outlier_mask]
y_clean = y[outlier_mask]

print(f"Data asli: {len(X_enhanced)} samples")
print(f"Setelah outlier removal: {len(X_clean)} samples")
print(f"Outliers removed: {len(X_enhanced) - len(X_clean)} samples")

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

# 4. SCALING - Gunakan RobustScaler untuk menangani outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. HYPERPARAMETER TUNING untuk Random Forest
print("Melakukan hyperparameter tuning...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

print(f"Best RF parameters: {rf_grid.best_params_}")
best_rf = rf_grid.best_estimator_

# 6. COBA MODEL ALTERNATIF - Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb = GradientBoostingRegressor(random_state=42)
gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gb_grid.fit(X_train_scaled, y_train)

print(f"Best GB parameters: {gb_grid.best_params_}")
best_gb = gb_grid.best_estimator_

# 7. ENSEMBLE MODEL - Kombinasi RF dan GB
class EnsembleModel:
    def __init__(self, rf_model, gb_model, rf_weight=0.5, gb_weight=0.5):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.rf_weight = rf_weight
        self.gb_weight = gb_weight
    
    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)
        return self.rf_weight * rf_pred + self.gb_weight * gb_pred

ensemble_model = EnsembleModel(best_rf, best_gb)

# 8. EVALUASI SEMUA MODEL
models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'Ensemble': ensemble_model
}

print("\n" + "="*50)
print("HASIL EVALUASI MODEL")
print("="*50)

best_model = None
best_mse = float('inf')

for name, model in models.items():
    # Cross-validation
    if name != 'Ensemble':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"\n{name}:")
        print(f"CV RMSE: {cv_rmse:.3f} ± {np.sqrt(cv_scores.std()):.3f}")
    
    # Prediksi pada test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE: {mae:.3f}")
    print(f"Test MSE: {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")
    
    # Tentukan model terbaik berdasarkan MSE
    if mse < best_mse:
        best_mse = mse
        best_model = (name, model)

# 9. ANALISIS FEATURE IMPORTANCE
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*50)

feature_names = list(X_clean.columns)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# 10. VISUALISASI HASIL
plt.figure(figsize=(15, 10))

# Plot 1: Feature Importance
plt.subplot(2, 3, 1)
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')

# Plot 2: Actual vs Predicted
y_pred_best = best_model[1].predict(X_test_scaled)

plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted ({best_model[0]})')

# Plot 3: Residuals
residuals = y_test - y_pred_best
plt.subplot(2, 3, 3)
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 4: Distribution of residuals
plt.subplot(2, 3, 4)
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

# Plot 5: Error distribution
plt.subplot(2, 3, 5)
errors = np.abs(residuals)
plt.hist(errors, bins=30, alpha=0.7)
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Distribution of Absolute Errors')

# Plot 6: Model comparison
plt.subplot(2, 3, 6)
model_names = list(models.keys())
mse_scores = [mean_squared_error(y_test, model.predict(X_test_scaled)) for model in models.values()]
plt.bar(model_names, mse_scores)
plt.title('Model Comparison (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 11. PREDIKSI UNTUK DATA BARU
print("\n" + "="*50)
print("CONTOH PREDIKSI UNTUK DATA BARU")
print("="*50)

# Contoh data baru
sample_data = pd.DataFrame({
    'Pizza Size': [2],
    'Pizza Type': [1],
    'Toppings Count': [3],
    'Distance (km)': [5.2],
    'Traffic Level': [3],
    'Topping Density': [0.8],
    'Order Month': [6],
    'Pizza Complexity': [2],
    'Order Hour': [19]
})

# Tambahkan engineered features
sample_data['Distance_Traffic'] = sample_data['Distance (km)'] * sample_data['Traffic Level']
sample_data['Complexity_Distance'] = sample_data['Pizza Complexity'] * sample_data['Distance (km)']
sample_data['Toppings_Density'] = sample_data['Toppings Count'] * sample_data['Topping Density']
sample_data['Is_Rush_Hour'] = int(((sample_data['Order Hour'].iloc[0] >= 11) & (sample_data['Order Hour'].iloc[0] <= 13)) | 
                                 ((sample_data['Order Hour'].iloc[0] >= 18) & (sample_data['Order Hour'].iloc[0] <= 20)))
sample_data['Is_Weekend'] = int((sample_data['Order Month'].iloc[0] == 6) | (sample_data['Order Month'].iloc[0] == 7))

# Scale dan prediksi
sample_scaled = scaler.transform(sample_data)
prediction = best_model[1].predict(sample_scaled)[0]

print(f"Prediksi delivery duration: {prediction:.2f} menit")
print(f"Model terbaik: {best_model[0]}")

# 12. SAVE MODEL DAN SCALER
print("\n" + "="*50)
print("MENYIMPAN MODEL DAN SCALER")
print("="*50)

# Fungsi untuk menyimpan model dan scaler
def save_model_and_scaler(model, scaler, model_path='best_model.pkl', scaler_path='scaler.pkl'):
    """
    Simpan model dan scaler untuk digunakan di Streamlit
    """
    # Simpan model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Simpan scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# Fungsi untuk menyimpan dengan joblib
def save_with_joblib(model, scaler, model_path='best_model.joblib', scaler_path='scaler.joblib'):
    """
    Simpan menggunakan joblib (recommended untuk sklearn)
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# Simpan model terbaik dan scaler
print(f"Menyimpan model terbaik: {best_model[0]}")
save_model_and_scaler(best_model[1], scaler)

# Simpan juga dengan joblib sebagai backup
save_with_joblib(best_model[1], scaler)

# Simpan juga feature names untuk referensi
feature_names_df = pd.DataFrame({'feature_names': list(X_clean.columns)})
feature_names_df.to_csv('feature_names.csv', index=False)
print("Feature names saved to feature_names.csv")

print("\n" + "="*50)
print("TRAINING SELESAI!")
print("="*50)
print("File yang dihasilkan:")
print("- best_model.pkl")
print("- scaler.pkl")
print("- best_model.joblib")
print("- scaler.joblib")
print("- feature_names.csv")
print("\nFile-file ini bisa digunakan untuk aplikasi Streamlit!")