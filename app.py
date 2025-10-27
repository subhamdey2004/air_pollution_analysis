from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from functools import wraps
import json
import secrets

app = Flask(__name__)

# Auto-generate SECRET_KEY if not provided
SECRET_KEY = os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///air_pollution_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Load the trained model and scaler
model = None
scaler = None

def load_models():
    """Load ML models safely"""
    global model, scaler
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'pollution_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully")
        else:
            print(f"⚠️  Model file not found at: {model_path}")
            print("   Create a test model or provide your trained model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure the model file is not corrupted")
        model = None
    
    try:
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully")
        else:
            print(f"⚠️  Scaler file not found at: {scaler_path}")
            print("   Create a test scaler or provide your trained scaler")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        print("   Make sure the scaler file is not corrupted")
        scaler = None

# Load models when app starts
load_models()

# ==================== DATABASE MODELS ====================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash password using bcrypt"""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    prediction_result = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'city': self.city,
            'input_data': self.input_data,
            'prediction_result': self.prediction_result,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


# ==================== DECORATORS ====================

def login_required(f):
    """Decorator to check if user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ==================== HELPER FUNCTIONS ====================

def get_quality_level(aqi_score):
    """Classify air quality based on AQI score"""
    if aqi_score < 50:
        return "Good"
    elif aqi_score < 100:
        return "Satisfactory"
    elif aqi_score < 200:
        return "Moderately Polluted"
    elif aqi_score < 300:
        return "Poor"
    else:
        return "Very Poor"


def validate_prediction_data(data):
    """Validate all required fields are present"""
    required_fields = ['city', 'PM25', 'PM10', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            return False, f"Missing field: {field}"
    
    # Validate city name
    if not isinstance(data['city'], str) or len(data['city'].strip()) == 0:
        return False, "City name must be a valid string"
    
    # Validate numeric fields
    try:
        for field in ['PM25', 'PM10', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']:
            value = float(data[field])
            if value < 0:
                return False, f"{field} cannot be negative"
    except (ValueError, TypeError):
        return False, "All numeric fields must be valid numbers"
    
    return True, "Valid"


# ==================== ROUTES - AUTHENTICATION ====================

@app.route('/')
def index():
    """Home page - redirect to login or dashboard"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')

        # Validation
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400

        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

        # Check if user exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already exists'}), 400

        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Registration successful! Please login.'}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Error creating user: {str(e)}'}), 500

    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email
            return jsonify({'success': True, 'message': 'Login successful'}), 200

        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('login'))


# ==================== ROUTES - DASHBOARD & PREDICTIONS ====================

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page"""
    user_id = session.get('user_id')
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).limit(5).all()
    return render_template('dashboard.html', predictions=predictions)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Make a prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, message = validate_prediction_data(data)
        if not is_valid:
            return jsonify({'success': False, 'message': message}), 400
        
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({'success': False, 'message': 'Model not available. Please check server logs.'}), 500
        
        city = data.get('city', '').strip()
        
        # Extract features in correct order (must match training)
        try:
            features = [
                float(data.get('PM25', 0)),
                float(data.get('PM10', 0)),
                float(data.get('NO2', 0)),
                float(data.get('NOx', 0)),
                float(data.get('NH3', 0)),
                float(data.get('CO', 0)),
                float(data.get('SO2', 0)),
                float(data.get('O3', 0))
            ]
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'Invalid numeric values'}), 400

        # Scale features
        try:
            features_array = np.array([features])
            features_scaled = scaler.transform(features_array)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error scaling features: {str(e)}'}), 500
        
        # Make prediction
        try:
            prediction = model.predict(features_scaled)[0]
            prediction = float(prediction)  # Ensure it's a float
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error making prediction: {str(e)}'}), 500
        
        # Classify quality
        quality = get_quality_level(prediction)
        
        # Save to database
        try:
            pred_record = Prediction(
                user_id=session.get('user_id'),
                city=city,
                input_data={
                    'PM25': float(data.get('PM25', 0)),
                    'PM10': float(data.get('PM10', 0)),
                    'NO2': float(data.get('NO2', 0)),
                    'NOx': float(data.get('NOx', 0)),
                    'NH3': float(data.get('NH3', 0)),
                    'CO': float(data.get('CO', 0)),
                    'SO2': float(data.get('SO2', 0)),
                    'O3': float(data.get('O3', 0))
                },
                prediction_result=prediction
            )
            db.session.add(pred_record)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Error saving prediction: {str(e)}'}), 500

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'quality': quality,
            'message': 'Prediction completed successfully'
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': f'Unexpected error: {str(e)}'}), 500


@app.route('/history')
@login_required
def history():
    """View prediction history"""
    user_id = session.get('user_id')
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)


# ==================== ROUTES - API ENDPOINTS ====================

@app.route('/api/prediction/<int:pred_id>')
@login_required
def get_prediction(pred_id):
    """Get specific prediction details"""
    user_id = session.get('user_id')
    prediction = Prediction.query.get(pred_id)
    
    if not prediction:
        return jsonify({'success': False, 'message': 'Prediction not found'}), 404
    
    if prediction.user_id != user_id:
        return jsonify({'success': False, 'message': 'Unauthorized access'}), 403
    
    return jsonify({
        'success': True,
        'id': prediction.id,
        'city': prediction.city,
        'input_data': prediction.input_data,
        'prediction_result': prediction.prediction_result,
        'created_at': prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')
    }), 200


@app.route('/api/user/profile')
@login_required
def get_user_profile():
    """Get current user profile"""
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    }), 200


@app.route('/api/predictions/stats')
@login_required
def get_prediction_stats():
    """Get prediction statistics for current user"""
    user_id = session.get('user_id')
    predictions = Prediction.query.filter_by(user_id=user_id).all()
    
    if not predictions:
        return jsonify({
            'success': True,
            'total_predictions': 0,
            'avg_aqi': 0,
            'max_aqi': 0,
            'min_aqi': 0
        }), 200
    
    aqi_scores = [p.prediction_result for p in predictions]
    
    return jsonify({
        'success': True,
        'total_predictions': len(predictions),
        'avg_aqi': round(np.mean(aqi_scores), 2),
        'max_aqi': round(np.max(aqi_scores), 2),
        'min_aqi': round(np.min(aqi_scores), 2)
    }), 200


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'message': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    db.session.rollback()
    return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ==================== CONTEXT PROCESSOR ====================

@app.context_processor
def inject_user():
    """Make user info available in all templates"""
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return {'current_user': user}
    return {'current_user': None}


# ==================== MAIN ====================

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")
    
    # Run the application
    print("\n" + "="*50)
    print("🌍 Air Pollution Analysis Application")
    print("="*50)
    print("Starting server...")
    print("Visit: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)