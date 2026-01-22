from flask import jsonify
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# -------- Database setup --------
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'stock.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------- Database models --------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(10), nullable=False)
    start_date = db.Column(db.String(20))
    end_date = db.Column(db.String(20))
    graph_path = db.Column(db.String(100))

# Create DB tables if not exist
with app.app_context():
    db.create_all()
    # Add default admin user if not exist
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', password='admin123')
        db.session.add(admin)
        db.session.commit()

# -------- Routes --------
@app.route('/history')
def history():
    all_predictions = Prediction.query.order_by(Prediction.id.desc()).all()
    return render_template('history.html', predictions=all_predictions)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username, password=password).first()
    if user:
        return redirect(url_for('predict'))
    else:
        return render_template('login.html', error='Invalid Credentials')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        company = request.form['company']
        start = request.form['startDate']
        end = request.form['endDate']

        # Download stock data
        symbol = company.upper().strip()

       # Auto handle Indian stocks
        if not symbol.endswith(('.NS', '.BO')) and symbol not in ['AAPL','MSFT','GOOG','AMZN']:
            symbol = symbol + '.NS'

        data = yf.download(symbol, start=start, end=end)

        if data.empty:
            return f"No data found for {symbol}. Try adding .NS for Indian stocks (Example: INFY.NS)"


        data.reset_index(inplace=True)
        data['DateOrdinal'] = data['Date'].map(pd.Timestamp.toordinal)
        X = data[['DateOrdinal']]
        y = data['Close']

        model = LinearRegression()
        model.fit(X, y)

        # Predict next 10 days
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=10)
        future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        predictions = model.predict(future_ordinals)

        # Save plot
        plt.figure(figsize=(10,5))
        plt.plot(data['Date'], y, label='Actual Price')
        plt.plot(future_dates, predictions, label='Predicted Price', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f"{company} Stock Price Prediction")
        plt.legend()

        # Replace old static graph save with unique filename
        timestamp = int(time.time())  # unique number for each prediction
        graph_filename = f'plot_{company}_{timestamp}.png'
        graph_path = os.path.join('static', graph_filename)
        plt.savefig(graph_path)
        plt.close()


        # Save prediction in DB
        new_pred = Prediction(company=company, start_date=start, end_date=end, graph_path=graph_path)
        db.session.add(new_pred)
        db.session.commit()

        return render_template('predict.html', image=graph_path)

    return render_template('predict.html', image=None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    company = data.get('company')
    start = data.get('start_date')
    end = data.get('end_date')

    if not company or not start or not end:
        return jsonify({'error': 'Please provide company, start_date, end_date'}), 400

    # Download stock data
    stock_data = yf.download(company, start=start, end=end)
    if stock_data.empty:
        return jsonify({'error': 'No data found for this company'}), 404

    stock_data.reset_index(inplace=True)
    stock_data['DateOrdinal'] = stock_data['Date'].map(pd.Timestamp.toordinal)
    X = stock_data[['DateOrdinal']]
    y = stock_data['Close']

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 10 days
    future_dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=10)
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    predictions = model.predict(future_ordinals)

    # Save plot with unique timestamp
    import time
    timestamp = int(time.time())
    graph_filename = f'plot_{company}_{timestamp}.png'
    graph_path = os.path.join('static', graph_filename)
    plt.figure(figsize=(10,5))
    plt.plot(stock_data['Date'], y, label='Actual Price')
    plt.plot(future_dates, predictions, label='Predicted Price', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f"{company} Stock Price Prediction")
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    # Save to database
    new_pred = Prediction(company=company, start_date=start, end_date=end, graph_path=graph_path)
    db.session.add(new_pred)
    db.session.commit()

    # Return JSON response
    response = {
        'company': company,
        'start_date': start,
        'end_date': end,
        'future_dates': [str(d.date()) for d in future_dates],
        'predictions': predictions.tolist(),
        'graph_url': url_for('static', filename=graph_filename, _external=True)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
