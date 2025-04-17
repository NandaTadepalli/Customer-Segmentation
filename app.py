from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from clustering import get_clustered_data

app = Flask(__name__)


def serialize_data(data):
    """Ensure all data is JSON serializable"""
    for point in data:
        for key, value in point.items():
            if isinstance(value, (np.int64, np.int32)):
                point[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                point[key] = float(value)
    return data


@app.route('/')
def index():
    try:
        # Get clustered data and ensure it's serializable
        points = get_clustered_data()
        points = serialize_data(points)
        return render_template('index.html', points=points)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
