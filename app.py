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
import re
import warnings
warnings.filterwarnings('ignore')

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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .warning-card {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #d4edda, #a8e6cf);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .danger-card {
        background: linear-gradient(135deg, #f8d7da, #ff7675);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
HUGGINGFACE_TOKEN = "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"
SUPABASE_URL = "https://dwwunwxkqtawcojrcrai.supabase.co"

# Initialize database
def init_database():
    """Initialize SQLite database with all required tables."""
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
        ai_risk_score INTEGER DEFAULT 0,
        follow_up_required BOOLEAN DEFAULT 0,
        outcome TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_by TEXT
    )
    """)

    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        email TEXT,
        full_name TEXT,
        active BOOLEAN DEFAULT 1,
        last_login DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_by TEXT
    )
    """)

    # Create chat history table for natural language queries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        query TEXT,
        response TEXT,
        query_type TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create custom reports table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS custom_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_name TEXT NOT NULL,
        report_config TEXT NOT NULL,
        created_by TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_modified DATETIME DEFAULT CURRENT_TIMESTAMP
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
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, email, full_name, created_by) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, ('admin', hash_password('AdminTemp!2025'), 'admin', 'admin@facility.org', 'System Administrator', 'system'))
        conn.commit()
    conn.close()

def get_user_by_username(username):
    """Get user details by username."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ? AND active = 1', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(username, password, role, email="", full_name="", created_by=""):
    """Create a new user."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, email, full_name, created_by) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, hash_password(password), role, email, full_name, created_by))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_users():
    """Get all users."""
    conn = sqlite3.connect('naloxone_incidents.db')
    df = pd.read_sql_query("""
        SELECT id, username, role, email, full_name, active, last_login, created_at, created_by
        FROM users ORDER BY created_at DESC
    """, conn)
    conn.close()
    return df

def update_user_status(user_id, active):
    """Update user active status."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET active = ? WHERE id = ?', (active, user_id))
    conn.commit()
    conn.close()

# AI Analysis Functions
def analyze_description_with_ai(description):
    """Enhanced AI analysis of incident descriptions."""
    if not description or len(description.strip()) < 10:
        return {"sentiment": "No analysis - insufficient text", "risk_score": 0}

    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"

        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": description[:500]},
            timeout=10
        )

        # Calculate risk score based on keywords
        high_risk_keywords = ['unresponsive', 'unconscious', 'cpr', 'blue', 'purple', 'not breathing', 'overdose', 'collapsed']
        medium_risk_keywords = ['dizzy', 'nauseous', 'confused', 'slow breathing', 'pale']

        risk_score = 0
        desc_lower = description.lower()

        for keyword in high_risk_keywords:
            if keyword in desc_lower:
                risk_score += 3

        for keyword in medium_risk_keywords:
            if keyword in desc_lower:
                risk_score += 1

        sentiment_result = "Analysis unavailable"
        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result, list) and len(result) > 0:
                best_result = max(result[0], key=lambda x: x['score'])
                sentiment = best_result['label']
                confidence = best_result['score']
                sentiment_result = f"{sentiment} (confidence: {confidence:.2f})"

        return {
            "sentiment": sentiment_result,
            "risk_score": min(risk_score, 10)  # Cap at 10
        }

    except Exception as e:
        return {"sentiment": f"Analysis error: {str(e)[:50]}", "risk_score": 0}

# Natural Language Query Processing
def process_natural_language_query(query, df):
    """Process natural language queries about the data."""
    query_lower = query.lower()

    try:
        # Common query patterns
        if any(word in query_lower for word in ['how many', 'count', 'total', 'number']):
            if 'incident' in query_lower:
                total_incidents = len(df)
                return f"There are **{total_incidents}** total incidents in the system."

            elif 'naloxone' in query_lower or 'dose' in query_lower:
                total_doses = df['nasal_naloxone'].sum() + df['intramuscular_naloxone'].sum()
                nasal_doses = df['nasal_naloxone'].sum()
                im_doses = df['intramuscular_naloxone'].sum()
                return f"Total naloxone doses administered: **{total_doses}** (Nasal: {nasal_doses}, Intramuscular: {im_doses})"

            elif 'location' in query_lower:
                unique_locations = df['location'].nunique()
                return f"There are **{unique_locations}** unique locations where incidents occurred."

        elif any(word in query_lower for word in ['most', 'highest', 'top']):
            if 'location' in query_lower:
                top_location = df['location'].value_counts().head(1)
                if not top_location.empty:
                    return f"The most common incident location is **{top_location.index[0]}** with **{top_location.values[0]}** incidents."

            elif 'common' in query_lower and 'member' in query_lower:
                top_member = df['community_member'].value_counts().head(1)
                if not top_member.empty:
                    return f"Community member with most incidents: **{top_member.index[0]}** with **{top_member.values[0]}** incidents."

        elif 'average' in query_lower or 'mean' in query_lower:
            if 'dose' in query_lower:
                avg_doses = (df['nasal_naloxone'].sum() + df['intramuscular_naloxone'].sum()) / len(df) if len(df) > 0 else 0
                return f"Average naloxone doses per incident: **{avg_doses:.2f}**"

        elif 'when' in query_lower or 'time' in query_lower:
            if 'last' in query_lower:
                if not df.empty:
                    latest_date = df['date'].max()
                    return f"The most recent incident was on **{latest_date}**."

            elif 'first' in query_lower:
                if not df.empty:
                    earliest_date = df['date'].min()
                    return f"The earliest incident in the system was on **{earliest_date}**."

        elif any(word in query_lower for word in ['high risk', 'dangerous', 'severe']):
            high_risk_incidents = df[df['ai_risk_score'] >= 6] if 'ai_risk_score' in df.columns else pd.DataFrame()
            if not high_risk_incidents.empty:
                return f"There are **{len(high_risk_incidents)}** high-risk incidents (risk score ≥ 6) requiring special attention."
            else:
                return "No high-risk incidents identified in the current data."

        elif 'staff' in query_lower:
            if 'most active' in query_lower or 'busiest' in query_lower:
                # Parse staff data
                all_staff = []
                for staff_list in df['staff_involved'].dropna():
                    if staff_list and str(staff_list).strip():
                        staff_names = [name.strip() for name in str(staff_list).split(',')]
                        all_staff.extend(staff_names)

                if all_staff:
                    staff_counts = pd.Series(all_staff).value_counts()
                    top_staff = staff_counts.head(1)
                    return f"Most active staff member: **{top_staff.index[0]}** involved in **{top_staff.values[0]}** incidents."

        elif 'trend' in query_lower or 'pattern' in query_lower:
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
                if len(monthly_counts) > 1:
                    recent_trend = monthly_counts.tail(3).values
                    if recent_trend[-1] > recent_trend[0]:
                        return f"**Increasing trend**: Incidents have increased over the last 3 months ({recent_trend[0]} → {recent_trend[-1]})."
                    elif recent_trend[-1] < recent_trend[0]:
                        return f"**Decreasing trend**: Incidents have decreased over the last 3 months ({recent_trend[0]} → {recent_trend[-1]})."
                    else:
                        return f"**Stable trend**: Incident numbers remain relatively stable over the last 3 months."

        # Default response with suggestions
        return f"""I'm not sure how to answer that specific question. Here are some questions you can ask:

**📊 Data Queries:**
- "How many total incidents are there?"
- "What's the most common location?"
- "How many naloxone doses were given?"
- "Who is the most active staff member?"

**📈 Analysis Queries:**
- "Show me the trends over time"
- "Which incidents are high risk?"
- "What's the average doses per incident?"
- "When was the last incident?"

**🔍 Specific Searches:**
- "Show incidents at [location name]"
- "Find incidents involving [person name]"
- "What happened on [date]?"
"""

    except Exception as e:
        return f"Sorry, I encountered an error processing your query: {str(e)}"

# Data Management Functions
def load_incidents_from_db():
    """Load incidents from SQLite database."""
    conn = sqlite3.connect('naloxone_incidents.db')
    try:
        df = pd.read_sql_query("""
        SELECT s_no, date, time, community_member, staff_involved, location,
               nasal_naloxone, intramuscular_naloxone, description, 
               source_sheet, ai_sentiment, ai_risk_score, follow_up_required,
               outcome, created_at, created_by
        FROM incidents ORDER BY date DESC, time DESC
        """, conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def save_incidents_to_db(df, created_by="system"):
    """Save incidents to SQLite database with enhanced AI analysis."""
    conn = sqlite3.connect('naloxone_incidents.db')

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df.iterrows():
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id FROM incidents WHERE s_no = ? AND date = ?',
            (str(row.get('s_no', '')), str(row.get('date', '')))
        )

        if not cursor.fetchone():
            status_text.text(f'Processing incident {idx + 1}/{len(df)} with AI analysis...')

            # Enhanced AI analysis
            ai_analysis = analyze_description_with_ai(str(row.get('description', '')))

            cursor.execute("""
            INSERT INTO incidents (
                s_no, date, time, community_member, staff_involved, location,
                nasal_naloxone, intramuscular_naloxone, description, 
                source_sheet, ai_sentiment, ai_risk_score, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ai_analysis['sentiment'],
                ai_analysis['risk_score'],
                created_by
            ))

        progress_bar.progress((idx + 1) / len(df))

    progress_bar.empty()
    status_text.empty()
    conn.commit()
    conn.close()

def process_csv_data(uploaded_file):
    """Process uploaded CSV file and return cleaned DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)

        # Column mapping for your specific CSV structure
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

        # Clean and convert data types
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        for col in ['nasal_naloxone', 'intramuscular_naloxone']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Fill NaN values
        df = df.fillna('')

        return df

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

# Enhanced Visualization Functions - FIXED VERSION
def create_advanced_visualizations(df):
    """Create advanced data visualizations - FIXED."""
    if df.empty:
        st.info("📊 No data available for visualizations.")
        return

    st.header("📈 Advanced Data Visualizations")

    # Convert date column safely
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df_valid = df.dropna(subset=['date'])
    except:
        df_valid = df

    viz_tabs = st.tabs(["📊 Overview", "🗺️ Geographic", "⏰ Temporal", "👥 Individual", "🚨 Risk Analysis"])

    with viz_tabs[0]:  # Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📈 Incident Trends")
            try:
                monthly_counts = df_valid.groupby(df_valid['date'].dt.to_period('M')).size()
                fig = px.line(
                    x=[str(p) for p in monthly_counts.index], 
                    y=monthly_counts.values,
                    title="Monthly Incident Trends",
                    labels={'x': 'Month', 'y': 'Number of Incidents'}
                )
                fig.update_traces(line=dict(width=3, color='#1f77b4'))
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("Unable to create trend chart - insufficient date data")

        with col2:
            st.subheader("💉 Naloxone Distribution")
            try:
                nasal_total = df['nasal_naloxone'].sum()
                im_total = df['intramuscular_naloxone'].sum()

                if nasal_total > 0 or im_total > 0:
                    fig = px.pie(
                        values=[nasal_total, im_total], 
                        names=['Nasal', 'Intramuscular'],
                        title="Naloxone Administration Methods"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No naloxone usage data available")
            except Exception as e:
                st.info("Unable to create naloxone chart")

        with col3:
            st.subheader("🎯 Risk Score Distribution")
            try:
                if 'ai_risk_score' in df.columns:
                    risk_data = df['ai_risk_score'].dropna()
                    if len(risk_data) > 0:
                        fig = px.histogram(
                            x=risk_data, 
                            title="AI Risk Score Distribution",
                            labels={'x': 'Risk Score', 'y': 'Frequency'},
                            nbins=10
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No risk score data available")
                else:
                    st.info("Risk scores not calculated yet")
            except Exception as e:
                st.info("Unable to create risk score chart")

    with viz_tabs[1]:  # Geographic
        st.subheader("🗺️ Geographic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            try:
                # Top locations
                location_counts = df['location'].value_counts().head(15)
                if not location_counts.empty:
                    fig = px.bar(
                        x=location_counts.values, 
                        y=location_counts.index, 
                        orientation='h',
                        title="Incident Hotspots by Location",
                        labels={'x': 'Number of Incidents', 'y': 'Location'},
                        color=location_counts.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No location data available")
            except Exception as e:
                st.info("Unable to create location chart")

        with col2:
            try:
                # Naloxone usage by location
                location_naloxone = df.groupby('location').agg({
                    'nasal_naloxone': 'sum',
                    'intramuscular_naloxone': 'sum'
                }).head(10)

                if not location_naloxone.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Nasal', 
                        x=location_naloxone.index, 
                        y=location_naloxone['nasal_naloxone'],
                        marker_color='#1f77b4'
                    ))
                    fig.add_trace(go.Bar(
                        name='Intramuscular', 
                        x=location_naloxone.index, 
                        y=location_naloxone['intramuscular_naloxone'],
                        marker_color='#ff7f0e'
                    ))
                    fig.update_layout(
                        title='Naloxone Usage by Location', 
                        barmode='stack', 
                        height=500,
                        xaxis_title='Location',
                        yaxis_title='Doses'
                    )
                    # FIXED: Use update_xaxes instead of update_xaxis
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No naloxone usage data by location")
            except Exception as e:
                st.info("Unable to create naloxone by location chart")

    with viz_tabs[2]:  # Temporal
        st.subheader("⏰ Temporal Patterns")

        col1, col2 = st.columns(2)

        with col1:
            try:
                # Day of week analysis
                if 'date' in df_valid.columns and not df_valid.empty:
                    df_valid['day_of_week'] = df_valid['date'].dt.day_name()
                    day_counts = df_valid['day_of_week'].value_counts().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    )
                    fig = px.bar(
                        x=day_counts.index, 
                        y=day_counts.values, 
                        title="Incidents by Day of Week",
                        labels={'x': 'Day', 'y': 'Number of Incidents'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No date data for day-of-week analysis")
            except Exception as e:
                st.info("Unable to create day-of-week chart")

        with col2:
            try:
                # Hour of day analysis (if time data available)
                if 'time' in df.columns:
                    time_hours = []
                    for time_str in df['time'].dropna():
                        try:
                            if ':' in str(time_str):
                                hour = int(str(time_str).split(':')[0])
                                if 0 <= hour <= 23:
                                    time_hours.append(hour)
                        except:
                            continue

                    if time_hours:
                        hourly_counts = pd.Series(time_hours).value_counts().sort_index()
                        fig = px.bar(
                            x=hourly_counts.index, 
                            y=hourly_counts.values, 
                            title="Incidents by Hour of Day",
                            labels={'x': 'Hour', 'y': 'Number of Incidents'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid time data for hourly analysis")
                else:
                    st.info("No time data available")
            except Exception as e:
                st.info("Unable to create hourly chart")

    with viz_tabs[3]:  # Individual Analysis
        st.subheader("👥 Individual Community Member Analysis")

        try:
            cm_counts = df['community_member'].value_counts()
            repeat_incidents = cm_counts[cm_counts > 1].head(10)

            if not repeat_incidents.empty:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        x=repeat_incidents.values, 
                        y=repeat_incidents.index, 
                        orientation='h',
                        title="Community Members with Multiple Incidents",
                        labels={'x': 'Number of Incidents', 'y': 'Community Member'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Individual risk progression
                    frequent_person = repeat_incidents.index[0]
                    person_data = df[df['community_member'] == frequent_person].sort_values('date')

                    if 'ai_risk_score' in person_data.columns and len(person_data) > 1:
                        fig = px.line(
                            person_data, 
                            x='date', 
                            y='ai_risk_score',
                            title=f"Risk Score Progression: {frequent_person}",
                            labels={'date': 'Date', 'ai_risk_score': 'Risk Score'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No risk progression data for {frequent_person}")
            else:
                st.info("No repeat incidents found")
        except Exception as e:
            st.info("Unable to create individual analysis charts")

    with viz_tabs[4]:  # Risk Analysis
        st.subheader("🚨 Risk Analysis Dashboard")

        try:
            if 'ai_risk_score' in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    # Risk vs Naloxone usage
                    df['total_naloxone'] = df['nasal_naloxone'] + df['intramuscular_naloxone']
                    risk_data = df[df['ai_risk_score'] > 0]

                    if not risk_data.empty:
                        fig = px.scatter(
                            risk_data, 
                            x='ai_risk_score', 
                            y='total_naloxone',
                            title="Risk Score vs Total Naloxone Usage",
                            labels={'ai_risk_score': 'Risk Score', 'total_naloxone': 'Total Naloxone Doses'},
                            trendline="ols"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No risk score data for correlation analysis")

                with col2:
                    # Risk distribution by location
                    risk_by_location = df.groupby('location')['ai_risk_score'].mean().sort_values(ascending=False).head(10)

                    if not risk_by_location.empty:
                        fig = px.bar(
                            x=risk_by_location.values, 
                            y=risk_by_location.index, 
                            orientation='h',
                            title="Average Risk Score by Location",
                            labels={'x': 'Average Risk Score', 'y': 'Location'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No risk score data by location")
            else:
                st.info("Risk scores not calculated yet - import data to see risk analysis")
        except Exception as e:
            st.info("Unable to create risk analysis charts")

# Custom Report Generation - Simplified Version
def create_custom_report_builder():
    """Custom report builder interface - simplified."""
    st.header("🏗️ Custom Report Builder")

    df = load_incidents_from_db()
    if df.empty:
        st.info("📊 No data available for custom reports. Please import data first.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Report Configuration")

        report_name = st.text_input("Report Name", placeholder="Enter report name")

        # Date range selection
        st.write("**Date Range:**")
        date_from = st.date_input("From Date", value=None)
        date_to = st.date_input("To Date", value=None)

        # Column selection
        available_columns = ['date', 'time', 'community_member', 'staff_involved', 'location', 
                           'nasal_naloxone', 'intramuscular_naloxone', 'description', 'ai_sentiment', 'ai_risk_score']
        available_columns = [col for col in available_columns if col in df.columns]
        selected_columns = st.multiselect("Select Columns to Include", available_columns, default=available_columns[:6])

        # Filters
        st.write("**Filters:**")
        location_filter = st.multiselect("Filter by Location", df['location'].unique())
        if 'ai_risk_score' in df.columns:
            risk_filter = st.slider("Minimum Risk Score", 0, 10, 0)
        else:
            risk_filter = 0

    with col2:
        st.subheader("👁️ Preview")

        # Apply filters
        filtered_df = df.copy()

        try:
            if date_from:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['date']) >= pd.Timestamp(date_from)]

            if date_to:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['date']) <= pd.Timestamp(date_to)]

            if location_filter:
                filtered_df = filtered_df[filtered_df['location'].isin(location_filter)]

            if 'ai_risk_score' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['ai_risk_score'] >= risk_filter]

            preview_df = filtered_df[selected_columns].head(10) if selected_columns else filtered_df.head(10)
            st.dataframe(preview_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error applying filters: {str(e)}")

        # Generate report
        if st.button("📊 Generate Custom Report", type="primary"):
            if report_name:
                try:
                    # Generate Excel report
                    excel_file = generate_custom_excel_report(filtered_df, selected_columns, report_name)
                    if excel_file:
                        st.success(f"✅ Custom report '{report_name}' generated successfully!")
                        st.download_button(
                            label="📥 Download Custom Report",
                            data=excel_file,
                            file_name=f"{report_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
            else:
                st.error("Please enter a report name.")

def generate_custom_excel_report(df, columns, report_name):
    """Generate custom Excel report."""
    if df.empty:
        return None

    try:
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main data sheet
            selected_df = df[columns] if columns else df
            selected_df.to_excel(writer, sheet_name='Report Data', index=False)

            # Summary sheet
            summary_data = {
                'Report Name': [report_name],
                'Generated On': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Total Records': [len(df)],
                'Columns Included': [len(columns) if columns else len(df.columns)]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Report Info', index=False)

        output.seek(0)
        return output

    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

# Chat interface for natural language queries
def create_chat_interface():
    """Create chat interface for natural language queries."""
    st.header("💬 Ask Questions About Your Data")

    df = load_incidents_from_db()

    if df.empty:
        st.info("💬 No data available for queries. Please import data first.")
        return

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Chat input
    user_query = st.chat_input("Ask a question about your incident data...")

    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Process the query
        response = process_natural_language_query(user_query, df)

        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Save to database
        try:
            save_chat_to_db(st.session_state.username, user_query, response)
        except:
            pass  # Don't fail if chat saving fails

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def save_chat_to_db(user_id, query, response):
    """Save chat interaction to database."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (user_id, query, response, query_type)
        VALUES (?, ?, ?, ?)
    """, (user_id, query, response, 'natural_language'))
    conn.commit()
    conn.close()

# User Management Interface - Simplified
def create_user_management():
    """Create user management interface for admins."""
    if st.session_state.role != 'admin':
        st.error("🚫 Access denied. Admin privileges required.")
        return

    st.header("👥 User Management")

    tab1, tab2 = st.tabs(["👨‍💼 All Users", "➕ Add User"])

    with tab1:
        st.subheader("Current Users")
        users_df = get_all_users()

        if not users_df.empty:
            for _, user in users_df.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.write(f"**{user['username']}** ({user['role']})")
                    if user['email']:
                        st.caption(f"📧 {user['email']}")

                with col2:
                    status = "🟢 Active" if user['active'] else "🔴 Inactive"
                    st.write(status)

                with col3:
                    if user['last_login']:
                        st.caption(f"Last: {str(user['last_login'])[:10]}")
                    else:
                        st.caption("Never logged in")

                with col4:
                    if user['username'] != 'admin':
                        new_status = not user['active']
                        action = "Activate" if new_status else "Deactivate"
                        if st.button(action, key=f"toggle_{user['id']}"):
                            update_user_status(user['id'], new_status)
                            st.rerun()

                st.divider()

    with tab2:
        st.subheader("➕ Add New User")

        with st.form("add_user_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_username = st.text_input("Username*", placeholder="Enter username")
                new_email = st.text_input("Email", placeholder="user@domain.com")
                new_role = st.selectbox("Role*", ["analyst", "viewer", "admin"])

            with col2:
                new_full_name = st.text_input("Full Name", placeholder="John Doe")
                auto_generate = st.checkbox("Auto-generate secure password", value=True)
                new_password = st.text_input("Password*", type="password", 
                                           placeholder="Leave blank for auto-generated password")

            submitted = st.form_submit_button("👤 Create User", type="primary")

            if submitted:
                if new_username:
                    try:
                        # Generate password if needed
                        if auto_generate or not new_password:
                            import secrets
                            import string
                            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
                            new_password = ''.join(secrets.choice(alphabet) for _ in range(12))

                        # Create user
                        if create_user(new_username, new_password, new_role, new_email, 
                                     new_full_name, st.session_state.username):
                            st.success(f"✅ User '{new_username}' created successfully!")
                            st.info(f"🔑 Password: `{new_password}`")
                            st.warning("⚠️ Please save this password securely. It won't be shown again.")
                        else:
                            st.error("❌ Failed to create user. Username may already exist.")
                    except Exception as e:
                        st.error(f"Error creating user: {str(e)}")
                else:
                    st.error("⚠️ Username is required.")

# Login page
def login_page():
    """Display clean login page."""
    st.markdown('<div class="main-header"><h1>🏥 Naloxone Response Management</h1><p>Professional Incident Tracking & Analytics Platform</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 Secure System Login")

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🚪 Sign In", use_container_width=True)

            if submitted:
                if username and password:
                    user = get_user_by_username(username)
                    if user and verify_password(password, user[2]):  # user[2] is password_hash
                        # Update last login
                        conn = sqlite3.connect('naloxone_incidents.db')
                        cursor = conn.cursor()
                        cursor.execute('UPDATE users SET last_login = ? WHERE username = ?', 
                                     (datetime.now().isoformat(), username))
                        conn.commit()
                        conn.close()

                        # Set session state
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = user[3]  # user[3] is role
                        st.session_state.user_id = user[0]  # user[0] is id
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("⚠️ Please enter both username and password")

# Main dashboard - Simplified
def main_dashboard():
    """Enhanced main dashboard - simplified."""
    # Header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<div class="main-header"><h1>🏥 Naloxone Response Management System</h1><p>Welcome back, <strong>{st.session_state.username}</strong> ({st.session_state.role.title()})</p></div>', unsafe_allow_html=True)

    with col2:
        if st.button("🔓 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Load data
    df = load_incidents_from_db()

    # Main navigation tabs
    if st.session_state.role == 'admin':
        tabs = st.tabs(["📊 Dashboard", "📁 Data Import", "📈 Analytics", "💬 AI Assistant", "🏗️ Reports", "👥 Users", "📋 Data View"])
    else:
        tabs = st.tabs(["📊 Dashboard", "📁 Data Import", "📈 Analytics", "💬 AI Assistant", "🏗️ Reports", "📋 Data View"])

    with tabs[0]:  # Dashboard
        if df.empty:
            st.info("📊 Welcome! Start by importing your incident data using the 'Data Import' tab.")

            # Quick start guide
            with st.expander("🚀 Quick Start Guide"):
                st.markdown("""
                **Getting Started:**
                1. 📁 Go to **Data Import** tab
                2. 📤 Upload your CSV file (drag & drop supported)
                3. 📊 View analytics in **Analytics** tab
                4. 💬 Ask questions using **AI Assistant**
                5. 🏗️ Create custom reports in **Reports** tab

                **Sample Questions for AI Assistant:**
                - "How many incidents were there last month?"
                - "What's the most common location?"
                - "Show me high-risk incidents"
                """)
        else:
            # Enhanced overview metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_incidents = len(df)
                st.markdown(f'<div class="metric-card"><h3>{total_incidents}</h3><p>Total Incidents</p></div>', unsafe_allow_html=True)

            with col2:
                total_doses = df['nasal_naloxone'].sum() + df['intramuscular_naloxone'].sum()
                st.markdown(f'<div class="metric-card"><h3>{total_doses}</h3><p>Naloxone Doses</p></div>', unsafe_allow_html=True)

            with col3:
                unique_people = df['community_member'].nunique()
                st.markdown(f'<div class="metric-card"><h3>{unique_people}</h3><p>Community Members</p></div>', unsafe_allow_html=True)

            with col4:
                unique_locations = df['location'].nunique()
                st.markdown(f'<div class="metric-card"><h3>{unique_locations}</h3><p>Locations</p></div>', unsafe_allow_html=True)

            with col5:
                if 'ai_risk_score' in df.columns:
                    high_risk = len(df[df['ai_risk_score'] >= 6])
                    st.markdown(f'<div class="danger-card"><h3>{high_risk}</h3><p>High Risk</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card"><h3>0</h3><p>High Risk</p></div>', unsafe_allow_html=True)

    with tabs[1]:  # Data Import
        st.header("📁 Data Import & Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📤 Upload Your Data")

            uploaded_file = st.file_uploader(
                "Choose your incident CSV file",
                type=['csv'],
                help="Upload your naloxone incident data. Supports drag & drop!"
            )

            if uploaded_file is not None:
                with st.spinner("🔄 Processing your data with AI analysis..."):
                    processed_df = process_csv_data(uploaded_file)

                    if processed_df is not None:
                        st.success(f"✅ Successfully processed {len(processed_df)} incidents!")

                        # Enhanced preview
                        st.subheader("👁️ Data Preview")
                        st.dataframe(processed_df.head(10), use_container_width=True)

                        # Import button
                        if st.button("💾 Import Data with AI Analysis", type="primary"):
                            try:
                                with st.spinner("🤖 Importing data and running AI analysis..."):
                                    save_incidents_to_db(processed_df, st.session_state.username)
                                    st.success("🎉 Data successfully imported with AI insights!")
                                    st.balloons()
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error importing data: {str(e)}")

        with col2:
            st.subheader("📊 Current Data Status")

            current_count = len(df)
            st.metric("📈 Total Incidents", current_count)

            if not df.empty:
                try:
                    date_range = f"{df['date'].min()} to {df['date'].max()}"
                    st.success(f"📅 **Date Range:** {date_range}")

                    unique_locations = df['location'].nunique()
                    st.info(f"📍 **Locations:** {unique_locations}")

                    unique_people = df['community_member'].nunique()
                    st.info(f"👥 **People:** {unique_people}")

                    if 'ai_risk_score' in df.columns:
                        avg_risk = df['ai_risk_score'].mean()
                        st.warning(f"🎯 **Avg Risk Score:** {avg_risk:.1f}/10")
                except:
                    st.info("Data statistics loading...")

    with tabs[2]:  # Analytics
        create_advanced_visualizations(df)

    with tabs[3]:  # AI Assistant
        create_chat_interface()

    with tabs[4]:  # Reports
        create_custom_report_builder()

    # User Management tab (admin only)
    if st.session_state.role == 'admin':
        with tabs[5]:
            create_user_management()

        with tabs[6]:  # Data View
            st.header("📋 Complete Data Management")

            if df.empty:
                st.info("📊 No data available. Import data first.")
            else:
                # Display data
                st.dataframe(df, use_container_width=True)
    else:
        with tabs[5]:  # Data View (non-admin)
            st.header("📋 Data View")

            if df.empty:
                st.info("📊 No data available.")
            else:
                st.dataframe(df, use_container_width=True)

# Main execution
if __name__ == "__main__":
    # Initialize database and create default admin
    init_database()
    create_default_admin()

    # Check login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Route to appropriate page
    if st.session_state.logged_in:
        main_dashboard()
    else:
        login_page()
