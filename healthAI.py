import streamlit as st
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict
import random
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
import hashlib
import re

load_dotenv()
# Helper: get secrets from Streamlit Cloud or fallback to local .env
def get_secret(key, default=None):
    try:
        import streamlit as st
        if key in st.secrets:  # Cloud
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)  # Local

# IBM Watsonx API configuration
WATSONX_URL = get_secret("WATSONX_URL")
WATSONX_APIKEY = get_secret("WATSONX_APIKEY")
WATSONX_SPACE_ID = get_secret("WATSONX_SPACE_ID")
WATSONX_MODEL_ID = get_secret("WATSONX_MODEL_ID")
if not WATSONX_APIKEY:
    st.error("⚠️ WATSONX_APIKEY is missing! Add it to Streamlit Secrets")
    st.stop()


# CSS Styling with better colors for chat readability
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%);
        padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;
    }
    .main-title {
        color: white !important; text-align: center; font-size: 3rem; font-weight: bold; margin-bottom: 0.5rem;
    }
    .main-subtitle {
        color: white !important; text-align: center; font-size: 1.2rem; opacity: 0.9;
    }
    .feature-card {
        background: #f5f5f7; color: #222; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; border-left: 4px solid #667eea;
    }
    .chat-message {
        padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: black !important; background: #fff !important;
    }
    .user-message {
        background-color: #cce5ff !important; border-left: 4px solid #2196f3; color: black !important;
    }
    .ai-message {
        background-color: #f8d7da !important; border-left: 4px solid #9c27b0; color: black !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;
    }
    .prediction-card {
        background: white; border: 2px solid #ff9800; border-radius: 10px;
        padding: 1.5rem; margin-bottom: 1rem; color: black !important;
    }
    .prediction-card p, .prediction-card li {
        color: black !important;
    }
    .prediction-card h1, .prediction-card h2, .prediction-card h3, .prediction-card h4 {
        color: #2c3e50 !important;
    }
    .treatment-section {
        background: #f8f9fa; border-radius: 10px;
        padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #28a745;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .health-tip {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;
    }
    .condition-card {
        background: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #4CAF50;
        color: black !important;
    }
    .recommendation-card {
        background: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #ffc107;
        color: black !important;
    }
    .symptom-input {
        background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
    }
    .analysis-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
    }
    /* Style the text area specifically */
    .stTextArea textarea {
        background-color: #f8f9fa !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    /* Style the placeholder text */
    .stTextArea textarea::placeholder {
        color: #666 !important;
        opacity: 0.8 !important;
    }
    /* Ensure all text in symptom section is visible */
    .symptom-input, .symptom-input h4, .symptom-input p {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Database Management
class DatabaseManager:
    def __init__(self, db_path="health_ai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                medical_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        # Health metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                metric_type TEXT,
                value REAL,
                unit TEXT,
                recorded_date DATE,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                symptoms TEXT,
                prediction TEXT,
                confidence REAL,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'patient',
                admin_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users (id)
            )
        """)
        # Admin-patient chat table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_patient_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER,
                patient_id INTEGER,
                sender_role TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users (id),
                FOREIGN KEY (patient_id) REFERENCES users (id)
            )
        """)
        conn.commit()
        conn.close()
    
    def add_patient(self, name, age, gender, contact, medical_history=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (name, age, gender, contact, medical_history)
            VALUES (?, ?, ?, ?, ?)
        """, (name, age, gender, contact, medical_history))
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return patient_id
    
    def get_all_patients(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY created_at DESC", conn)
        conn.close()
        return df
    
    def add_chat_message(self, patient_id, message, response):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (patient_id, message, response)
            VALUES (?, ?, ?)
        """, (patient_id, message, response))
        conn.commit()
        conn.close()
    
    def get_chat_history(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT message, response, timestamp 
            FROM chat_history 
            WHERE patient_id = ? 
            ORDER BY timestamp ASC
        """, conn, params=(patient_id,))
        conn.close()
        return df
    
    def add_health_metric(self, patient_id, metric_type, value, unit, recorded_date):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO health_metrics (patient_id, metric_type, value, unit, recorded_date)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, metric_type, value, unit, recorded_date))
        conn.commit()
        conn.close()
    
    def get_health_metrics(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM health_metrics 
            WHERE patient_id = ? 
            ORDER BY recorded_date DESC
        """, conn, params=(patient_id,))
        conn.close()
        return df
    
    def add_prediction(self, patient_id, symptoms, prediction, confidence, recommendations):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (patient_id, symptoms, prediction, confidence, recommendations)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, symptoms, prediction, confidence, recommendations))
        conn.commit()
        conn.close()
    
    def get_predictions(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY created_at DESC
        """, conn, params=(patient_id,))
        conn.close()
        return df
    
    def add_user(self, username, password, role='patient', admin_id=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, admin_id)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, role, admin_id))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def validate_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("""
            SELECT * FROM users WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        user = cursor.fetchone()
        conn.close()
        return user is not None
    
    def get_user(self, username):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        return user

    def get_admins(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE role = 'admin'")
        admins = cursor.fetchall()
        conn.close()
        return admins

    def get_patients_for_admin(self, admin_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE role = 'patient' AND admin_id = ?", (admin_id,))
        patients = cursor.fetchall()
        conn.close()
        return patients

    def add_admin_patient_message(self, admin_id, patient_id, sender_role, message):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO admin_patient_chat (admin_id, patient_id, sender_role, message)
            VALUES (?, ?, ?, ?)
        """, (admin_id, patient_id, sender_role, message))
        conn.commit()
        conn.close()

    def get_admin_patient_chat(self, admin_id, patient_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sender_role, message, timestamp FROM admin_patient_chat
            WHERE admin_id = ? AND patient_id = ?
            ORDER BY timestamp ASC
        """, (admin_id, patient_id))
        chat = cursor.fetchall()
        conn.close()
        return chat

# AI Integration Class
class HealthAI:
    def __init__(self):
        try:
            self.llm = WatsonxLLM(
                model_id=WATSONX_MODEL_ID,
                url=WATSONX_URL,
                apikey=WATSONX_APIKEY,
                space_id=WATSONX_SPACE_ID,
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens": 500,
                    "temperature": 0.7
                }
            )
            self.chat_template = self._create_chat_template()
            self.prediction_template = self._create_prediction_template()
        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")
            self.llm = None

    def _create_chat_template(self):
        return PromptTemplate(
            input_variables=["user_question", "chat_history"],
            template="""You are a Health AI assistant providing medical information and guidance.

IMPORTANT: You are not a replacement for professional medical advice. Always recommend consulting healthcare professionals for serious concerns.

Previous conversation context:
{chat_history}

Current question: {user_question}

Provide helpful, accurate health information while being empathetic and clear. Include relevant health tips and recommendations for a healthy lifestyle.

Response:"""
        )

    def _create_prediction_template(self):
        return PromptTemplate(
            input_variables=["symptoms", "patient_info"],
            template="""You are a medical AI assistant analyzing symptoms to provide potential health insights.

Patient Information: {patient_info}
Reported Symptoms: {symptoms}

Based on the symptoms provided, analyze and provide:
1. Most likely conditions (with confidence levels)
2. Recommended immediate actions
3. When to seek medical attention
4. Lifestyle recommendations

IMPORTANT: This is for informational purposes only. Always recommend consulting a healthcare professional for proper diagnosis.

Please format your response in a clear, structured way with headings for each section.

Response:"""
        )

    def generate_chat_response(self, message: str, chat_history: str = "") -> str:
        if not self.llm:
            return "AI service is currently unavailable. Please try again later."

        try:
            prompt = self.chat_template.format(
                user_question=message,
                chat_history=chat_history
            )
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Sorry, there was an error processing your request: {str(e)}"

    def predict_condition(self, symptoms: str, patient_info: str) -> str:
        if not self.llm:
            return "AI service is currently unavailable. Please try again later."

        try:
            prompt = self.prediction_template.format(
                symptoms=symptoms,
                patient_info=patient_info
            )
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Sorry, there was an error processing your request: {str(e)}"

# Data Generation Utilities
def generate_sample_health_data(patient_id: int, days: int = 30):
    """Generate sample health metrics for demonstration"""
    db = DatabaseManager()

    metrics = [
        {"type": "blood_pressure_systolic", "unit": "mmHg", "range": (110, 140)},
        {"type": "blood_pressure_diastolic", "unit": "mmHg", "range": (70, 90)},
        {"type": "heart_rate", "unit": "bpm", "range": (60, 100)},
        {"type": "weight", "unit": "kg", "range": (60, 90)},
        {"type": "blood_sugar", "unit": "mg/dL", "range": (80, 120)},
        {"type": "temperature", "unit": "°F", "range": (97, 99)}
    ]

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        for metric in metrics:
            value = round(random.uniform(metric["range"][0], metric["range"][1]), 1)
            db.add_health_metric(patient_id, metric["type"], value, metric["unit"], date)

def login_signup_page():
    st.title("Health AI - Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if st.session_state.db.validate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    user = st.session_state.db.get_user(username)
                    st.session_state.user_role = user[3] if user else 'patient'
                    st.session_state.user_id = user[0] if user else None
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["patient", "admin"])
            admin_id = None
            
            if role == "patient":
                admins = st.session_state.db.get_admins()
                if admins:
                    admin_choices = []
                    admin_id_map = {}
                    
                    for admin in admins:
                        admin_display = f"{admin[1]} (ID: {admin[0]})"
                        admin_choices.append(admin_display)
                        admin_id_map[admin_display] = admin[0]
                    
                    selected_admin = st.selectbox("Select Admin", admin_choices)
                    admin_id = admin_id_map[selected_admin]
                else:
                    st.warning("No admins available. Please create an admin account first.")
                    
            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if role == "patient" and not admin_id:
                        st.error("Please select an admin for patient account")
                    else:
                        success = st.session_state.db.add_user(username, password, role, admin_id)
                        if success:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username already exists")

def show_home_page():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Welcome to Health AI</h3>
            <p>Your comprehensive healthcare assistant powered by IBM Granite AI. Get personalized health insights, chat with AI, predict potential conditions, and manage your health data all in one place.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h4>💬 AI Chat Assistant</h4>
            <p>Ask health questions and get intelligent responses</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h4>🔮 Disease Prediction</h4>
            <p>Analyze symptoms to predict potential conditions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Health Analytics</h4>
            <p>Visualize and track your health metrics</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="health-tip">
            <h4>💡 Daily Health Tip</h4>
            <p>Stay hydrated! Aim for 8 glasses of water daily to maintain optimal health and energy levels.</p>
        </div>
        """, unsafe_allow_html=True)

def show_chat_assistant():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()
    st.header("💬 Health Chat Assistant")
    
    # For simplicity, we'll use the current user as the patient
    patient_id = st.session_state.user_id
    
    st.subheader(f"Chat with Health AI")
    chat_history = st.session_state.db.get_chat_history(patient_id)
    chat_container = st.container()
    with chat_container:
        if not chat_history.empty:
            for _, row in chat_history.iterrows():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {row['message']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>Health AI:</strong> {row['response']}
                </div>
                """, unsafe_allow_html=True)
    user_message = st.text_input("Ask a health question:", key="chat_input")
    if st.button("Send") and user_message:
        with st.spinner("AI is thinking..."):
            chat_context = ""
            if not chat_history.empty:
                recent_chats = chat_history.tail(3)
                chat_context = "\\n".join([
                    f"User: {row['message']}\\nAI: {row['response']}" 
                    for _, row in recent_chats.iterrows()])
            ai_response = st.session_state.ai.generate_chat_response(user_message, chat_context)
            st.session_state.db.add_chat_message(patient_id, user_message, ai_response)
            st.rerun()

def show_disease_prediction():
    if 'ai' not in st.session_state or st.session_state.ai is None:
        st.session_state.ai = HealthAI()
    st.header("🔮 Disease Prediction System")
    
    # For simplicity, we'll use the current user as the patient
    patient_id = st.session_state.user_id
    
    # Get user info for patient details
    user_info = st.session_state.db.get_user(st.session_state.username)
    patient_details = f"User ID: {user_info[0]}, Username: {user_info[1]}, Role: {user_info[3]}"

    st.subheader("Symptom Analysis")
    
    # Symptom input section with better styling
    st.markdown("""
    <div class="symptom-input">
        <h4>Describe your symptoms</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symptoms = st.text_area(
            "Enter details:",
            placeholder="e.g., Fever for 3 days, headache, body aches, fatigue...",
            height=120,
            key="symptom_input"
        )
        
        if st.button("Analyze Symptoms", type="primary", use_container_width=True):
            if symptoms:
                with st.spinner("Analyzing symptoms..."):
                    prediction_result = st.session_state.ai.predict_condition(symptoms, patient_details)
                    
                    if "error" not in prediction_result.lower():
                        # Store the prediction
                        st.session_state.db.add_prediction(
                            patient_id, symptoms,
                            prediction_result,
                            0.7,  # Default confidence
                            "See analysis above"
                        )
                        st.success("Analysis Complete!")
                        
                        # Display the results with proper styling
                        st.subheader("Analysis Results")
                        st.markdown(f"""
                        <div class="prediction-card">
                            {prediction_result}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {prediction_result}")
            else:
                st.warning("Please describe your symptoms.")
    
    with col2:
        st.markdown("""
        <div class="analysis-result">
            <h4>📋 User Information</h4>
            <p><strong>Username:</strong> {}</p>
            <p><strong>Role:</strong> {}</p>
        </div>
        """.format(user_info[1], user_info[3]), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="health-tip">
            <h4>💡 Tips for Better Analysis</h4>
            <ul>
            <li>Be specific about symptom duration</li>
            <li>Note any triggers or patterns</li>
            <li>Mention all symptoms, even minor ones</li>
            <li>Include any existing conditions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Previous Predictions")
    predictions_df = st.session_state.db.get_predictions(patient_id)
    if not predictions_df.empty:
        for _, pred in predictions_df.iterrows():
            with st.expander(f"Prediction from {pred['created_at'][:10]}"):
                st.markdown(f"""
                <div class="condition-card">
                    <strong>Symptoms:</strong> {pred['symptoms']}
                </div>
                <div class="recommendation-card">
                    <strong>Analysis:</strong> {pred['prediction']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No previous predictions found.")

def show_health_analytics():
    st.header("📊 Health Analytics")
    
    # For simplicity, we'll use the current user as the patient
    patient_id = st.session_state.user_id
    
    metrics_df = st.session_state.db.get_health_metrics(patient_id)
    if metrics_df.empty:
        st.info("No health metrics found for this user.")
        # Option to generate sample data
        if st.button("Generate Sample Health Data"):
            generate_sample_health_data(patient_id)
            st.success("Sample health data generated!")
            st.rerun()
        return
    
    st.subheader("Health Metric Trends")
    metric_types = metrics_df['metric_type'].unique()
    metric_selected = st.selectbox("Select metric to visualize", metric_types)
    metric_data = metrics_df[metrics_df['metric_type'] == metric_selected]
    fig = px.line(
        metric_data,
        x="recorded_date",
        y="value",
        title=f"{metric_selected.capitalize()} over Time",
        labels={"recorded_date": "Date", "value": metric_selected.capitalize()}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_report_analysis():
    st.header("📑 Report Analysis")
    st.info("This feature is under development. Future updates will allow upload and analysis of PDF and image-based health reports with AI.")

def show_admin_patient_chat():
    st.header("🗨️ Admin-Patient Chat")
    if st.session_state.get("user_role") != "admin":
        st.warning("Only admin users can access this page.")
        return
    patients = st.session_state.db.get_patients_for_admin(st.session_state.user_id)
    if not patients:
        st.info("No patients assigned to you.")
        return
    patient_options = {f"{p[1]} (ID: {p[0]})": p[0] for p in patients}
    selected_patient = st.selectbox("Select patient to chat with", list(patient_options.keys()))
    patient_id = patient_options[selected_patient]
    chat = st.session_state.db.get_admin_patient_chat(st.session_state.user_id, patient_id)
    st.subheader(f"Chat with {selected_patient.split(' (')[0]}")
    chat_container = st.container()
    with chat_container:
        for msg in chat:
            sender = "You" if msg[0] == "admin" else "Patient"
            st.markdown(f"**{sender}:** {msg[1]}  \n*{msg[2]}*")
    message = st.text_input("Your message to the patient:")
    if st.button("Send Message") and message:
        st.session_state.db.add_admin_patient_message(
            st.session_state.user_id, patient_id, "admin", message
        )
        st.success("Message sent.")
        st.rerun()

def main():
    st.set_page_config(
        page_title="Health AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css()
    
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    if 'ai' not in st.session_state:
        st.session_state.ai = HealthAI()
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login_signup_page()
        return
        
    if st.session_state.get("user_role") == "admin":
        st.sidebar.markdown(
            f"""
            <div style="background:#f3e5f5;padding:1em;border-radius:8px;margin-bottom:1em;">
                <strong>Your Admin ID:</strong> <span style="color:#6a1b9a;font-size:1.2em;">{st.session_state.get("user_id")}</span><br>
                <small>Share this ID with patients for signup.</small>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏥 Health AI</h1>
        <p class="main-subtitle">Your Intelligent Healthcare Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    
    # Simplified navigation without Patient Management and Treatment Plans
    if st.session_state.get("user_role") == "admin":
        page_options = ["🏠 Home", "💬 Chat Assistant", "🔮 Disease Prediction", "📊 Health Analytics", "📑 Report Analysis", "🗨️ Admin-Patient Chat"]
    else:
        page_options = ["🏠 Home", "💬 Chat Assistant", "🔮 Disease Prediction", "📊 Health Analytics", "📑 Report Analysis"]
    
    page = st.sidebar.selectbox("Choose a page:", page_options)
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "💬 Chat Assistant":
        show_chat_assistant()
    elif page == "🔮 Disease Prediction":
        show_disease_prediction()
    elif page == "📊 Health Analytics":
        show_health_analytics()
    elif page == "📑 Report Analysis":
        show_report_analysis()
    elif page == "🗨️ Admin-Patient Chat":
        show_admin_patient_chat()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

if __name__ == "__main__":
    main()