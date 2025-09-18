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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import seaborn as sns
import matplotlib.pyplot as plt
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f4;
        border-radius: 8px 8px 0px 0px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
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

# Custom Report Generation
def create_custom_report_builder():
    """Custom report builder interface."""
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
        date_from = st.date_input("From Date")
        date_to = st.date_input("To Date")

        # Column selection
        available_columns = ['date', 'time', 'community_member', 'staff_involved', 'location', 
                           'nasal_naloxone', 'intramuscular_naloxone', 'description', 'ai_sentiment', 'ai_risk_score']
        selected_columns = st.multiselect("Select Columns to Include", available_columns, default=available_columns[:6])

        # Filters
        st.write("**Filters:**")
        location_filter = st.multiselect("Filter by Location", df['location'].unique())
        risk_filter = st.slider("Minimum Risk Score", 0, 10, 0)

        # Grouping and aggregation
        group_by = st.selectbox("Group By", ['None', 'location', 'community_member', 'staff_involved', 'date'])

        if group_by != 'None':
            agg_functions = st.multiselect("Aggregation Functions", ['count', 'sum', 'mean', 'max', 'min'])

    with col2:
        st.subheader("👁️ Preview")

        # Apply filters
        filtered_df = df.copy()

        if date_from:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['date']) >= pd.Timestamp(date_from)]

        if date_to:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['date']) <= pd.Timestamp(date_to)]

        if location_filter:
            filtered_df = filtered_df[filtered_df['location'].isin(location_filter)]

        if 'ai_risk_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ai_risk_score'] >= risk_filter]

        # Apply grouping
        if group_by != 'None' and 'agg_functions' in locals() and agg_functions:
            if group_by in filtered_df.columns:
                numeric_cols = ['nasal_naloxone', 'intramuscular_naloxone', 'ai_risk_score']
                agg_dict = {}

                for func in agg_functions:
                    if func in ['count']:
                        agg_dict['date'] = 'count'
                    else:
                        for col in numeric_cols:
                            if col in filtered_df.columns:
                                agg_dict[col] = func

                if agg_dict:
                    preview_df = filtered_df.groupby(group_by).agg(agg_dict).reset_index()
                else:
                    preview_df = filtered_df[selected_columns].head(10)
            else:
                preview_df = filtered_df[selected_columns].head(10)
        else:
            preview_df = filtered_df[selected_columns].head(10)

        st.dataframe(preview_df, use_container_width=True)

        # Generate report
        if st.button("📊 Generate Custom Report", type="primary"):
            if report_name:
                # Create report configuration
                report_config = {
                    'name': report_name,
                    'date_from': str(date_from) if date_from else None,
                    'date_to': str(date_to) if date_to else None,
                    'columns': selected_columns,
                    'location_filter': location_filter,
                    'risk_filter': risk_filter,
                    'group_by': group_by,
                    'agg_functions': agg_functions if group_by != 'None' and 'agg_functions' in locals() else []
                }

                # Save report configuration
                save_custom_report(report_name, report_config, st.session_state.username)

                # Generate Excel report
                excel_file = generate_custom_excel_report(filtered_df, report_config)
                if excel_file:
                    st.success(f"✅ Custom report '{report_name}' generated successfully!")
                    st.download_button(
                        label="📥 Download Custom Report",
                        data=excel_file,
                        file_name=f"{report_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("Please enter a report name.")

def save_custom_report(report_name, report_config, created_by):
    """Save custom report configuration."""
    conn = sqlite3.connect('naloxone_incidents.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO custom_reports (report_name, report_config, created_by)
        VALUES (?, ?, ?)
    """, (report_name, json.dumps(report_config), created_by))
    conn.commit()
    conn.close()

def get_saved_reports():
    """Get all saved custom reports."""
    conn = sqlite3.connect('naloxone_incidents.db')
    df = pd.read_sql_query("""
        SELECT id, report_name, created_by, created_at, last_modified
        FROM custom_reports ORDER BY created_at DESC
    """, conn)
    conn.close()
    return df

def generate_custom_excel_report(df, config):
    """Generate custom Excel report based on configuration."""
    if df.empty:
        return None

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main data sheet
        if config.get('group_by') != 'None' and config.get('agg_functions'):
            # Grouped data
            group_by = config['group_by']
            numeric_cols = ['nasal_naloxone', 'intramuscular_naloxone', 'ai_risk_score']
            agg_dict = {}

            for func in config['agg_functions']:
                if func == 'count':
                    agg_dict['date'] = 'count'
                else:
                    for col in numeric_cols:
                        if col in df.columns:
                            agg_dict[col] = func

            if agg_dict and group_by in df.columns:
                grouped_df = df.groupby(group_by).agg(agg_dict).reset_index()
                grouped_df.to_excel(writer, sheet_name='Grouped Analysis', index=False)

        # Original data
        selected_df = df[config['columns']] if config['columns'] else df
        selected_df.to_excel(writer, sheet_name='Raw Data', index=False)

        # Summary sheet
        summary_data = {
            'Report Name': [config['name']],
            'Generated On': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Total Records': [len(df)],
            'Date Range': [f"{config.get('date_from', 'All')} to {config.get('date_to', 'All')}"],
            'Filters Applied': [f"Location: {config.get('location_filter', 'None')}, Risk ≥ {config.get('risk_filter', 0)}"]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Report Info', index=False)

    output.seek(0)
    return output

# Enhanced Visualization Functions
def create_advanced_visualizations(df):
    """Create advanced data visualizations."""
    if df.empty:
        st.info("📊 No data available for visualizations.")
        return

    st.header("📈 Advanced Data Visualizations")

    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_valid = df.dropna(subset=['date'])

    viz_tabs = st.tabs(["📊 Overview", "🗺️ Geographic", "⏰ Temporal", "👥 Individual", "🚨 Risk Analysis", "📋 Comparative"])

    with viz_tabs[0]:  # Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📈 Incident Trends")
            monthly_counts = df_valid.groupby(df_valid['date'].dt.to_period('M')).size()
            fig = px.line(x=[str(p) for p in monthly_counts.index], y=monthly_counts.values,
                         title="Monthly Incident Trends")
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💉 Naloxone Distribution")
            nasal_total = df['nasal_naloxone'].sum()
            im_total = df['intramuscular_naloxone'].sum()
            fig = px.pie(values=[nasal_total, im_total], names=['Nasal', 'Intramuscular'],
                        title="Naloxone Administration Methods")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("🎯 Risk Score Distribution")
            if 'ai_risk_score' in df.columns:
                fig = px.histogram(df, x='ai_risk_score', title="AI Risk Score Distribution",
                                 labels={'ai_risk_score': 'Risk Score', 'count': 'Frequency'})
                st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[1]:  # Geographic
        st.subheader("🗺️ Geographic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Top locations heatmap-style
            location_counts = df['location'].value_counts().head(15)
            fig = px.bar(x=location_counts.values, y=location_counts.index, orientation='h',
                        title="Incident Hotspots by Location", color=location_counts.values,
                        color_continuous_scale='Reds')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Naloxone usage by location
            location_naloxone = df.groupby('location').agg({
                'nasal_naloxone': 'sum',
                'intramuscular_naloxone': 'sum'
            }).head(10)

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Nasal', x=location_naloxone.index, y=location_naloxone['nasal_naloxone']))
            fig.add_trace(go.Bar(name='Intramuscular', x=location_naloxone.index, y=location_naloxone['intramuscular_naloxone']))
            fig.update_layout(title='Naloxone Usage by Location', barmode='stack', height=500)
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[2]:  # Temporal
        st.subheader("⏰ Temporal Patterns")

        col1, col2 = st.columns(2)

        with col1:
            # Day of week analysis
            df_valid['day_of_week'] = df_valid['date'].dt.day_name()
            day_counts = df_valid['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig = px.bar(x=day_counts.index, y=day_counts.values, title="Incidents by Day of Week")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
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
                    fig = px.bar(x=hourly_counts.index, y=hourly_counts.values, title="Incidents by Hour of Day")
                    st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[3]:  # Individual Analysis
        st.subheader("👥 Individual Community Member Analysis")

        cm_counts = df['community_member'].value_counts()
        repeat_incidents = cm_counts[cm_counts > 1].head(10)

        if not repeat_incidents.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(x=repeat_incidents.values, y=repeat_incidents.index, orientation='h',
                           title="Community Members with Multiple Incidents")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Individual risk progression
                frequent_person = repeat_incidents.index[0]
                person_data = df[df['community_member'] == frequent_person].sort_values('date')

                if 'ai_risk_score' in person_data.columns and len(person_data) > 1:
                    fig = px.line(person_data, x='date', y='ai_risk_score',
                                 title=f"Risk Score Progression: {frequent_person}")
                    st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[4]:  # Risk Analysis
        st.subheader("🚨 Risk Analysis Dashboard")

        if 'ai_risk_score' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                # Risk vs Naloxone usage
                df['total_naloxone'] = df['nasal_naloxone'] + df['intramuscular_naloxone']
                fig = px.scatter(df, x='ai_risk_score', y='total_naloxone',
                               title="Risk Score vs Total Naloxone Usage",
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Risk distribution by location
                risk_by_location = df.groupby('location')['ai_risk_score'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=risk_by_location.values, y=risk_by_location.index, orientation='h',
                           title="Average Risk Score by Location")
                st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[5]:  # Comparative
        st.subheader("📋 Comparative Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Before/After comparison (if applicable)
            st.subheader("📊 Monthly Comparison")
            monthly_stats = df_valid.groupby(df_valid['date'].dt.to_period('M')).agg({
                'nasal_naloxone': 'sum',
                'intramuscular_naloxone': 'sum',
                'ai_risk_score': 'mean'
            }).tail(6)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=[str(p) for p in monthly_stats.index], y=monthly_stats['nasal_naloxone'], name="Nasal Doses"),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=[str(p) for p in monthly_stats.index], y=monthly_stats['ai_risk_score'], name="Avg Risk Score"),
                secondary_y=True
            )
            fig.update_yaxes(title_text="Naloxone Doses", secondary_y=False)
            fig.update_yaxes(title_text="Average Risk Score", secondary_y=True)
            fig.update_layout(title="Doses vs Risk Over Time")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Staff efficiency comparison
            st.subheader("👨‍⚕️ Staff Performance")
            all_staff = []
            for staff_list in df['staff_involved'].dropna():
                if staff_list and str(staff_list).strip():
                    staff_names = [name.strip() for name in str(staff_list).split(',')]
                    all_staff.extend(staff_names)

            if all_staff:
                staff_counts = pd.Series(all_staff).value_counts().head(8)
                fig = px.bar(x=staff_counts.values, y=staff_counts.index, orientation='h',
                           title="Staff Response Frequency")
                st.plotly_chart(fig, use_container_width=True)

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
        save_chat_to_db(st.session_state.username, user_query, response)

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

# User Management Interface
def create_user_management():
    """Create user management interface for admins."""
    if st.session_state.role != 'admin':
        st.error("🚫 Access denied. Admin privileges required.")
        return

    st.header("👥 User Management")

    tab1, tab2, tab3 = st.tabs(["👨‍💼 All Users", "➕ Add User", "📊 User Activity"])

    with tab1:
        st.subheader("Current Users")
        users_df = get_all_users()

        if not users_df.empty:
            # Display users with action buttons
            for _, user in users_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                with col1:
                    st.write(f"**{user['username']}** ({user['role']})")
                    if user['email']:
                        st.caption(f"📧 {user['email']}")

                with col2:
                    status = "🟢 Active" if user['active'] else "🔴 Inactive"
                    st.write(status)

                with col3:
                    if user['last_login']:
                        st.caption(f"Last: {user['last_login'][:10]}")
                    else:
                        st.caption("Never logged in")

                with col4:
                    if user['username'] != 'admin':  # Don't allow deactivating admin
                        new_status = not user['active']
                        action = "Activate" if new_status else "Deactivate"
                        if st.button(action, key=f"toggle_{user['id']}"):
                            update_user_status(user['id'], new_status)
                            st.rerun()

                with col5:
                    if user['username'] != st.session_state.username:  # Don't allow deleting own account
                        if st.button("🗑️", key=f"delete_{user['id']}", help="Delete User"):
                            # Add confirmation logic here
                            st.warning("Delete functionality requires confirmation dialog")

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
                new_password = st.text_input("Password*", type="password", 
                                           placeholder="Leave blank for auto-generated password")
                auto_generate = st.checkbox("Auto-generate secure password", value=True)

            submitted = st.form_submit_button("👤 Create User", type="primary")

            if submitted:
                if new_username:
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
                else:
                    st.error("⚠️ Username is required.")

    with tab3:
        st.subheader("📊 User Activity Overview")

        # Chat activity
        conn = sqlite3.connect('naloxone_incidents.db')
        try:
            chat_activity = pd.read_sql_query("""
                SELECT user_id, COUNT(*) as query_count, MAX(created_at) as last_query
                FROM chat_history 
                GROUP BY user_id 
                ORDER BY query_count DESC
            """, conn)

            if not chat_activity.empty:
                st.subheader("💬 Chat Query Activity")
                st.dataframe(chat_activity, use_container_width=True)
        except:
            st.info("No chat activity data available.")
        finally:
            conn.close()

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

# Main dashboard
def main_dashboard():
    """Enhanced main dashboard."""
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

            # Quick insights
            st.subheader("📊 Quick Insights")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Top location
                top_location = df['location'].value_counts().head(1)
                if not top_location.empty:
                    st.info(f"🏢 **Most Common Location:** {top_location.index[0]} ({top_location.values[0]} incidents)")

            with col2:
                # Recent trend
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                current_month = df[df['date'].dt.month == datetime.now().month]
                prev_month = df[df['date'].dt.month == datetime.now().month - 1]

                if not current_month.empty and not prev_month.empty:
                    change = len(current_month) - len(prev_month)
                    trend = "📈 Increasing" if change > 0 else "📉 Decreasing" if change < 0 else "➡️ Stable"
                    st.info(f"📅 **Monthly Trend:** {trend} ({change:+d})")

            with col3:
                # High-risk alerts
                if 'ai_risk_score' in df.columns:
                    high_risk_incidents = df[df['ai_risk_score'] >= 8]
                    if not high_risk_incidents.empty:
                        st.warning(f"⚠️ **Critical Alert:** {len(high_risk_incidents)} very high-risk incidents need immediate attention!")

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
                            with st.spinner("🤖 Importing data and running AI analysis..."):
                                save_incidents_to_db(processed_df, st.session_state.username)
                                st.success("🎉 Data successfully imported with AI insights!")
                                st.balloons()
                                st.rerun()

        with col2:
            st.subheader("📊 Current Data Status")

            current_count = len(df)
            st.metric("📈 Total Incidents", current_count)

            if not df.empty:
                date_range = f"{df['date'].min()} to {df['date'].max()}"
                st.success(f"📅 **Date Range:** {date_range}")

                unique_locations = df['location'].nunique()
                st.info(f"📍 **Locations:** {unique_locations}")

                unique_people = df['community_member'].nunique()
                st.info(f"👥 **People:** {unique_people}")

                if 'ai_risk_score' in df.columns:
                    avg_risk = df['ai_risk_score'].mean()
                    st.warning(f"🎯 **Avg Risk Score:** {avg_risk:.1f}/10")

    with tabs[2]:  # Analytics
        create_advanced_visualizations(df)

    with tabs[3]:  # AI Assistant
        create_chat_interface()

    with tabs[4]:  # Reports
        create_custom_report_builder()

        st.divider()

        # Quick export options
        if not df.empty:
            st.subheader("🚀 Quick Exports")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Generate Excel Report", use_container_width=True):
                    excel_data = generate_custom_excel_report(df, {'name': 'Quick Export', 'columns': df.columns.tolist()})
                    st.download_button(
                        "📥 Download Excel",
                        excel_data,
                        f"naloxone_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            with col2:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    f"naloxone_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col3:
                # Summary report
                summary = f"""NALOXONE INCIDENT SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
User: {st.session_state.username}

OVERVIEW:
• Total Incidents: {len(df)}
• Date Range: {df['date'].min()} to {df['date'].max()}
• Total Naloxone Doses: {df['nasal_naloxone'].sum() + df['intramuscular_naloxone'].sum()}
• Unique Locations: {df['location'].nunique()}
• Community Members: {df['community_member'].nunique()}

TOP LOCATIONS:
{df['location'].value_counts().head().to_string()}

HIGH-RISK INCIDENTS:
{len(df[df['ai_risk_score'] >= 6]) if 'ai_risk_score' in df.columns else 'Not calculated'}
"""
                st.download_button(
                    "📋 Download Summary",
                    summary,
                    f"naloxone_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain",
                    use_container_width=True
                )

    # User Management tab (admin only)
    if st.session_state.role == 'admin':
        with tabs[5]:
            create_user_management()

        with tabs[6]:  # Data View
            st.header("📋 Complete Data Management")

            if df.empty:
                st.info("📊 No data available. Import data first.")
            else:
                # Advanced filtering
                with st.expander("🔍 Advanced Filters"):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        location_filter = st.multiselect("Filter by Location", df['location'].unique())

                    with col2:
                        date_from = st.date_input("From Date")

                    with col3:
                        date_to = st.date_input("To Date")

                    with col4:
                        if 'ai_risk_score' in df.columns:
                            risk_filter = st.slider("Minimum Risk Score", 0, 10, 0)
                        else:
                            risk_filter = 0

                # Apply filters
                filtered_df = df.copy()

                if location_filter:
                    filtered_df = filtered_df[filtered_df['location'].isin(location_filter)]

                if date_from:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['date']) >= pd.Timestamp(date_from)]

                if date_to:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['date']) <= pd.Timestamp(date_to)]

                if 'ai_risk_score' in filtered_df.columns and risk_filter > 0:
                    filtered_df = filtered_df[filtered_df['ai_risk_score'] >= risk_filter]

                st.subheader(f"📊 Showing {len(filtered_df)} of {len(df)} incidents")

                # Display data with enhanced formatting
                display_columns = ['date', 'time', 'community_member', 'location', 'staff_involved', 
                                 'nasal_naloxone', 'intramuscular_naloxone', 'ai_sentiment', 'ai_risk_score']
                available_columns = [col for col in display_columns if col in filtered_df.columns]

                st.dataframe(
                    filtered_df[available_columns].sort_values('date', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        with tabs[5]:  # Data View (non-admin)
            st.header("📋 Data View")

            if df.empty:
                st.info("📊 No data available.")
            else:
                # Basic filtering for non-admin users
                col1, col2 = st.columns(2)

                with col1:
                    location_filter = st.selectbox("Filter by Location", ['All'] + sorted(df['location'].unique().tolist()))

                with col2:
                    date_filter = st.date_input("Filter by Date")

                # Apply filters
                filtered_df = df.copy()

                if location_filter != 'All':
                    filtered_df = filtered_df[filtered_df['location'] == location_filter]

                if date_filter:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['date']).dt.date == date_filter]

                st.dataframe(filtered_df, use_container_width=True)

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
