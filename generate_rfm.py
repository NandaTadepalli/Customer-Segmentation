import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Load the CSV files
orders = pd.read_csv('static/data/olist_orders_dataset.csv')
payments = pd.read_csv('static/data/olist_order_payments_dataset.csv')
customers = pd.read_csv('static/data/olist_customers_dataset.csv')

# Merge orders and customers
orders_customers = orders.merge(customers, on='customer_id')
full_data = orders_customers.merge(payments, on='order_id')

# Convert to datetime
full_data['order_purchase_timestamp'] = pd.to_datetime(full_data['order_purchase_timestamp'])
reference_date = full_data['order_purchase_timestamp'].max()

# Calculate RFM
rfm = full_data.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
    'order_id': 'nunique',
    'payment_value': 'sum'
}).reset_index()

rfm.columns = ['customer_unique_id', 'recency_days', 'frequency', 'monetary']

# Normalize RFM
scaler = StandardScaler()
scaled = scaler.fit_transform(rfm[['recency_days', 'frequency', 'monetary']])

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['kmeans_cluster'] = kmeans.fit_predict(scaled)

# PCA for visualization
pca = PCA(n_components=2)
rfm[['x', 'y']] = pca.fit_transform(scaled)

# SAMPLE for DBSCAN to avoid memory issues
sampled_rfm = rfm.sample(n=5000, random_state=42).reset_index(drop=True)
sampled_scaled = scaler.fit_transform(sampled_rfm[['recency_days', 'frequency', 'monetary']])

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
sampled_rfm['dbscan_cluster'] = dbscan.fit_predict(sampled_scaled)

# Fill DBSCAN clusters
rfm['dbscan_cluster'] = -1
rfm.loc[sampled_rfm.index, 'dbscan_cluster'] = sampled_rfm['dbscan_cluster']

# NOW merge customer information
customer_info = customers[['customer_unique_id', 'customer_city', 'customer_state']]
rfm = rfm.merge(customer_info, on='customer_unique_id', how='left')

# Save final output
rfm.to_csv('static/data/rfm_clustered.csv', index=False)

print("✅ Saved static/data/rfm_clustered.csv correctly with customer details!")
