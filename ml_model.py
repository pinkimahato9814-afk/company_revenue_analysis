from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

class MLModule:
    def __init__(self, merged_data, visualizer):
        self.df = merged_data
        self.viz = visualizer

    def run_all_ml(self):
        print("\n--- Starting Machine Learning Tasks ---\n")
        self.customer_segmentation()
        self.revenue_prediction()
        self.anomaly_detection()
        self.demand_forecasting()

    def customer_segmentation(self):
        print("ML: Performing Customer Segmentation (KMeans)...")
        # Feature engineering for RFM-like segmentation
        cust_data = self.df.groupby('customer_id').agg({
            'net_revenue': 'sum',
            'transaction_id': 'count',
            'date': lambda x: (pd.to_datetime(config.END_DATE) - x.max()).days
        }).rename(columns={'net_revenue': 'Monetary', 'transaction_id': 'Frequency', 'date': 'Recency'})
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cust_data)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cust_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        print("Customer Clusters Summary:")
        print(cust_data.groupby('cluster').mean())
        
        # Plotting clusters
        self.viz.plot_scatter(cust_data, 'Frequency', 'Monetary', 'Customer Segments: Frequency vs Monetary', 'ml_customer_clusters.png', hue='cluster')

    def revenue_prediction(self):
        print("ML: Training Revenue Prediction Model (Random Forest)...")
        # Prepare features
        ml_df = self.df.copy()
        ml_df['day_of_week'] = ml_df['date'].dt.dayofweek
        ml_df['month'] = ml_df['date'].dt.month
        
        # Encode categorical
        ml_df = pd.get_dummies(ml_df, columns=['category', 'region', 'payment_method'], drop_first=True)
        
        features = ['quantity', 'price', 'discount_pct', 'day_of_week', 'month'] + [col for col in ml_df.columns if col.startswith(('category_', 'region_', 'payment_method_'))]
        X = ml_df[features]
        y = ml_df['net_revenue']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"Revenue Prediction - R2 Score: {r2:.4f}, MSE: {mse:.2f}")
        
        # Plot feature importance
        importances = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False).head(10)
        self.viz.plot_bar(importances, 'feature', 'importance', 'Top 10 Features for Revenue Prediction', 'ml_feature_importance.png', horizontal=True)

    def anomaly_detection(self):
        print("ML: Detecting Anomalies in Sales (Isolation Forest)...")
        # Focus on revenue and quantity
        features = ['net_revenue', 'quantity']
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        self.df['anomaly'] = iso_forest.fit_predict(self.df[features])
        
        anomalies = self.df[self.df['anomaly'] == -1]
        print(f"Detected {len(anomalies)} potential anomalies in transactions.")
        
        self.viz.plot_scatter(self.df, 'quantity', 'net_revenue', 'Sales Anomaly Detection', 'ml_anomalies.png', hue='anomaly')

    def demand_forecasting(self):
        print("ML: Performing Basic Demand Forecasting...")
        # Daily revenue
        daily_rev = self.df.groupby('date')['net_revenue'].sum().reset_index()
        daily_rev['date_ordinal'] = daily_rev['date'].apply(lambda x: x.toordinal())
        
        X = daily_rev[['date_ordinal']]
        y = daily_rev['net_revenue']
        
        model = LinearRegression()
        model.fit(X, y)
        
        daily_rev['forecast'] = model.predict(X)
        
        print("Demand Forecast: Trend line calculated.")
        
        # Plot trend
        plt.figure(figsize=(12, 6))
        plt.plot(daily_rev['date'], daily_rev['net_revenue'], label='Actual Revenue', alpha=0.5)
        plt.plot(daily_rev['date'], daily_rev['forecast'], label='Forecast Trend', color='red', linewidth=2)
        plt.title('Daily Revenue Forecast Trend', fontsize=15)
        plt.legend()
        plt.savefig(os.path.join(config.PLOTS_DIR, "ml_demand_forecast.png"))
        plt.close()
