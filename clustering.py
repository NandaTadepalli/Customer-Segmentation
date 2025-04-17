import pandas as pd

def load_rfm():
    df = pd.read_csv('static/data/rfm_clustered.csv')
    return df

def get_clustered_data():
    df = load_rfm()
    points = []
    for _, row in df.iterrows():
        points.append({
            'x': row['x'],
            'y': row['y'],
            'kmeans': int(row['kmeans_cluster']),
            'dbscan': int(row['dbscan_cluster']),
            'recency': row['recency_days'],
            'frequency': row['frequency'],
            'monetary': row['monetary'],
            'customer_unique_id': row['customer_unique_id'],
            'city': row['customer_city'],
            'state': row['customer_state']
        })
    return points
