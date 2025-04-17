# Customer Segmentation Analysis Dashboard

This project implements a customer segmentation analysis dashboard using RFM (Recency, Frequency, Monetary) analysis and machine learning clustering techniques (K-means and DBSCAN) on the Brazilian E-commerce Olist dataset.

## Features

- RFM Analysis (Recency, Frequency, Monetary)
- K-means Clustering
- DBSCAN Clustering
- Interactive Visualizations:
  - Cluster Distribution
  - Customer Geographic Distribution
  - Purchase Patterns
  - Monetary Value Analysis
  - Customer Behavior Radar Charts

## Installation

1. Clone the repository:

```bash
git clone https://github.com/NandaTadepalli/Customer-Segmentation-.git
cd Customer-Segmentation-
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Data Setup

Place the following Olist dataset files in the `static/data` directory:

- olist_customers_dataset.csv
- olist_orders_dataset.csv
- olist_order_payments_dataset.csv

## Usage

1. Generate RFM analysis and clustering:

```bash
python generate_rfm.py
```

2. Run the Flask application:

```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Visualizations

- **Cluster Analysis**: View K-means and DBSCAN clustering results
- **Geographic Distribution**: See customer distribution across Brazilian states
- **Customer Behavior**: Analyze purchase patterns and customer value
- **Detailed Statistics**: Access detailed metrics for each customer segment

## Tech Stack

- Python
- Flask
- Pandas
- Scikit-learn
- Plotly.js
- Bootstrap 5

## License

This project is open source and available under the MIT License.

## Contributors

- Nanda Tadepalli
