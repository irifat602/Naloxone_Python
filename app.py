import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import requests
import json
import hashlib
import sqlite3
import base64

# Page configuration
st.set_page_config(
    page_title="🏥 Naloxone Incident Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3d59, #2b5876);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
HUGGINGFACE_TOKEN = "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"
SUPABASE_URL = "https://dwwunwxkqtawcojrcrai.supabase.co"

# Initialize database
def init_database():
    """Initialize SQLite database for storing incidents."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()

    # Create incidents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        s_no TEXT,
        date DATE,
        time TEXT,
        community_member TEXT,
        staff_involved TEXT,
        location TEXT,
        nasal_naloxone INTEGER DEFAULT 0,
        intramuscular_naloxone INTEGER DEFAULT 0,
        description TEXT,
        source_sheet TEXT,
        ai_sentiment TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

# Authentication functions
def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify a password against its hash."""
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def create_default_admin():
    """Create default admin user if not exists."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        cursor.execute(
            'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
            ('admin', hash_password('AdminTemp!2025'), 'admin')
        )
        conn.commit()
    conn.close()

# AI Analysis Functions
def analyze_description_with_ai(description):
    """Analyze incident description using HuggingFace API."""
    if not description or len(description.strip()) < 10:
        return "No analysis - insufficient text"

    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"

        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": description[:500]},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result, list) and len(result) > 0:
                best_result = max(result[0], key=lambda x: x['score'])
                sentiment = best_result['label']
                confidence = best_result['score']
                return f"{sentiment} (confidence: {confidence:.2f})"

        return "Analysis unavailable"

    except Exception as e:
        return f"Analysis error: {str(e)[:50]}"

# Data Management Functions
def load_incidents_from_db():
    """Load incidents from SQLite database."""
    conn = sqlite3.connect('naloxone_incidents.db')
    try:
        df = pd.read_sql_query("""
        SELECT s_no, date, time, community_member, staff_involved, location,
               nasal_naloxone, intramuscular_naloxone, description, 
               source_sheet, ai_sentiment, created_at
        FROM incidents ORDER BY date DESC, time DESC
        """, conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def save_incidents_to_db(df):
    """Save incidents to SQLite database."""
    conn = sqlite3.connect('naloxone_incidents.db')

    for _, row in df.iterrows():
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id FROM incidents WHERE s_no = ? AND date = ?',
            (str(row.get('s_no', '')), str(row.get('date', '')))
        )

        if not cursor.fetchone():
            ai_sentiment = analyze_description_with_ai(str(row.get('description', '')))

            cursor.execute("""
            INSERT INTO incidents (
                s_no, date, time, community_member, staff_involved, location,
                nasal_naloxone, intramuscular_naloxone, description, 
                source_sheet, ai_sentiment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get('s_no', '')),
                str(row.get('date', '')),
                str(row.get('time', '')),
                str(row.get('community_member', '')),
                str(row.get('staff_involved', '')),
                str(row.get('location', '')),
                int(row.get('nasal_naloxone', 0) or 0),
                int(row.get('intramuscular_naloxone', 0) or 0),
                str(row.get('description', '')),
                str(row.get('source_sheet', '')),
                ai_sentiment
            ))

    conn.commit()
    conn.close()

def process_csv_data(uploaded_file):
    """Process uploaded CSV file and return cleaned DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)

        # Column mapping
        column_mapping = {
            'S.No': 's_no',
            'Date': 'date',
            'Time': 'time',
            'Community Member(s) Involved': 'community_member',
            'Staff Involved': 'staff_involved',
            'Location': 'location',
            'Nasal Naloxone Administered': 'nasal_naloxone',
            'Intramuscular Naloxone Administered': 'intramuscular_naloxone',
            'Description of Incident': 'description',
            'Source Sheet': 'source_sheet'
        }

        df = df.rename(columns=column_mapping)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        for col in ['nasal_naloxone', 'intramuscular_naloxone']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        df = df.fillna('')
        return df

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

# Login page
def login_page():
    """Display login page."""
    st.markdown('<div class="main-header"><h1>🏥 Naloxone Response Management</h1><p>Secure Incident Tracking System</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 System Login")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if username and password:
                    conn = sqlite3.connect('naloxone_incidents.db')
                    cursor = conn.cursor()
                    cursor.execute('SELECT password_hash, role FROM users WHERE username = ?', (username,))
                    user = cursor.fetchone()
                    conn.close()

                    if user and verify_password(password, user[0]):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = user[1]
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("⚠️ Please enter both username and password")

# Main dashboard
def main_dashboard():
    """Main dashboard application."""
    st.markdown('<div class="main-header"><h1>🏥 Naloxone Response Management System</h1><p>Welcome back, ' + st.session_state.username + ' (' + st.session_state.role.title() + ')</p></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Navigation")

        if st.button("🔓 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Load data
    df = load_incidents_from_db()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📁 Data Import", "📋 Data View"])

    with tab1:
        st.header("📊 System Overview")

        if df.empty:
            st.info("📊 No data available. Please import your CSV data first.")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Incidents", len(df))

            with col2:
                total_doses = df['nasal_naloxone'].sum() + df['intramuscular_naloxone'].sum()
                st.metric("Total Naloxone Doses", total_doses)

            with col3:
                unique_people = df['community_member'].nunique()
                st.metric("Community Members", unique_people)

            with col4:
                unique_locations = df['location'].nunique()
                st.metric("Unique Locations", unique_locations)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top Locations")
                location_counts = df['location'].value_counts().head(10)
                fig = px.bar(x=location_counts.values, y=location_counts.index, orientation='h')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Naloxone Usage")
                nasal_total = df['nasal_naloxone'].sum()
                im_total = df['intramuscular_naloxone'].sum()
                fig = px.pie(values=[nasal_total, im_total], names=['Nasal', 'Intramuscular'])
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📁 Data Import")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            with st.spinner("🔄 Processing CSV file..."):
                processed_df = process_csv_data(uploaded_file)

                if processed_df is not None:
                    st.success(f"✅ Successfully processed {len(processed_df)} incidents!")

                    # Preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(processed_df.head(), use_container_width=True)

                    if st.button("💾 Import Data", type="primary"):
                        with st.spinner("💾 Importing data with AI analysis..."):
                            save_incidents_to_db(processed_df)
                            st.success("✅ Data imported successfully!")
                            st.rerun()

    with tab3:
        st.header("👥 Complete Data View")

        if df.empty:
            st.info("📊 No data available. Please import data first.")
        else:
            st.dataframe(df, use_container_width=True)

            # Download buttons
            col1, col2 = st.columns(2)

            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"naloxone_incidents_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

            with col2:
                # Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Incidents', index=False)
                output.seek(0)

                st.download_button(
                    "📊 Download Excel",
                    output,
                    f"naloxone_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Main execution
if __name__ == "__main__":
    init_database()
    create_default_admin()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_dashboard()
    else:
        login_page()
