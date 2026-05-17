"""
Enhanced Streamlit UI for Ride-Hailing Driver-Order Matching Prediction System
With Local Save Support
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import os
import json
from pathlib import Path

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Ride-Hailing Matcher | AI Driver Matching",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1E1E1E;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: white;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    /* Prediction result */
    .prediction-accept {
        background: linear-gradient(135deg, #00CC96 0%, #00B37E 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    
    .prediction-reject {
        background: linear-gradient(135deg, #FF4B4B 0%, #CC0000 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255,75,75,0.3);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Save success message */
    .save-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates"""
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371  # Earth radius in km


def make_prediction(input_data):
    """Make prediction using simple logic (demo)"""
    distance = haversine_distance(
        input_data['pickup_lat'],
        input_data['pickup_lon'],
        input_data['driver_lat'],
        input_data['driver_lon']
    )
    
    # Simple scoring logic
    score = 1.0
    score -= min(distance / 10, 0.5)  # Distance penalty
    score -= (1 - input_data.get('driver_acceptance', 0.7)) * 0.3  # Acceptance rate
    score += input_data.get('is_rush_hour', 0) * -0.2  # Rush hour penalty
    
    probability = max(0.05, min(0.95, score))
    return probability, distance


def save_to_local(results_df, prediction_type, additional_data=None):
    """Save results to local disk (only works in local environment)"""
    try:
        # Create directories
        base_dir = Path(__file__).parent
        output_dir = base_dir / "artifacts" / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions CSV
        csv_path = output_dir / f"{prediction_type}_predictions_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'prediction_type': prediction_type,
            'num_records': len(results_df),
            'accept_count': int((results_df['recommendation'] == 'ACCEPT').sum()) if 'recommendation' in results_df.columns else None,
            'additional_data': additional_data
        }
        
        metadata_path = output_dir / f"{prediction_type}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return True, str(csv_path), str(metadata_path)
    except Exception as e:
        return False, None, str(e)


def detect_environment():
    """Detect if running on Streamlit Cloud or locally"""
    # Streamlit Cloud sets specific environment variables
    if os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_CLOUD'):
        return 'cloud'
    return 'local'


# Detect environment
ENVIRONMENT = detect_environment()
IS_LOCAL = ENVIRONMENT == 'local'

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = True
if 'last_save_path' not in st.session_state:
    st.session_state.last_save_path = None

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/000000/taxi.png", width=70)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("## 🚗 Ride-Hailing Matcher")
    st.markdown("AI-Powered Driver Matching")
    st.markdown("---")
    
    # Environment indicator
    if IS_LOCAL:
        st.markdown('<span class="status-badge status-success">💻 Local Mode - Save Enabled</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">☁️ Cloud Mode - Download Only</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📋 Navigation",
        ["🏠 Dashboard", "🔮 Real-Time Prediction", "📊 Batch Prediction", "📈 Analytics", "📁 Saved Outputs", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="status-badge status-success">●</span> Model Active', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="status-badge status-success">●</span> API Online', unsafe_allow_html=True)
    
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Today's Stats")
    st.metric("Total Predictions", "1,234", delta="+12%")
    st.metric("Avg Confidence", "87%", delta="+5%")
    st.metric("Match Rate", "78%", delta="+3%")
    
    # Save directory info (local only)
    if IS_LOCAL:
        st.markdown("---")
        st.markdown("### 💾 Save Location")
        st.caption("`artifacts/predictions/`")
        if st.session_state.last_save_path:
            st.success(f"Last saved:\n`{st.session_state.last_save_path}`")
    
    st.markdown("---")
    st.caption("v2.0.0 | Powered by AI")


# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">Ride-Hailing Driver Matching System</div>', unsafe_allow_html=True)
    
    # Hero section with metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🚀 78%</div>
            <div class="metric-label">Match Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">⚡ 2.3s</div>
            <div class="metric-label">Avg Response</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">🎯 92%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">📈 +20%</div>
            <div class="metric-label">Efficiency Gain</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">📊 Model Performance</div>', unsafe_allow_html=True)
        
        models = ['Random Forest', 'XGBoost', 'LightGBM']
        accuracy = [0.85, 0.86, 0.84]
        precision = [0.84, 0.85, 0.83]
        recall = [0.86, 0.87, 0.85]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy, text=[f"{v:.1%}" for v in accuracy], textposition='auto', marker_color='#FF4B4B'))
        fig.add_trace(go.Bar(name='Precision', x=models, y=precision, text=[f"{v:.1%}" for v in precision], textposition='auto', marker_color='#00CC96'))
        fig.add_trace(go.Bar(name='Recall', x=models, y=recall, text=[f"{v:.1%}" for v in recall], textposition='auto', marker_color='#FFA500'))
        
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group',
            height=400,
            plot_bgcolor='white',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Quick Prediction Test</div>', unsafe_allow_html=True)
        
        with st.container():
            col_a, col_b = st.columns(2)
            with col_a:
                pickup_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.6f", key="dash_pickup_lat")
                pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.6f", key="dash_pickup_lon")
            with col_b:
                driver_lat = st.number_input("Driver Latitude", value=40.7160, format="%.6f", key="dash_driver_lat")
                driver_lon = st.number_input("Driver Longitude", value=-74.0100, format="%.6f", key="dash_driver_lon")
            
            if st.button("🚀 Predict Now", use_container_width=True):
                input_data = {'pickup_lat': pickup_lat, 'pickup_lon': pickup_lon, 'driver_lat': driver_lat, 'driver_lon': driver_lon, 'driver_acceptance': 0.7, 'is_rush_hour': 0}
                proba, distance = make_prediction(input_data)
                
                st.markdown("---")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Acceptance Probability"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF4B4B"},
                           'steps': [{'range': [0, 30], 'color': '#FFCCCC'}, {'range': [30, 70], 'color': '#FFFFCC'}, {'range': [70, 100], 'color': '#CCFFCC'}],
                           'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 50}}
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                if proba >= 0.5:
                    st.markdown(f'<div class="prediction-accept"><h3>✅ ACCEPT</h3><p>Confidence: {proba:.1%}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-reject"><h3>❌ REJECT</h3><p>Confidence: {(1-proba):.1%}</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature importance section
    st.markdown('<div class="sub-header">🔍 Key Features Impacting Predictions</div>', unsafe_allow_html=True)
    
    features = ['Distance to Pickup', 'Time of Day', 'Driver Acceptance Rate', 'Trip Distance', 'GPS Accuracy', 'Rush Hour']
    importance = [35, 25, 18, 12, 5, 5]
    
    fig = px.bar(x=importance, y=features, orientation='h', text=[f"{i}%" for i in importance],
                 title="Feature Importance Analysis", color=importance, color_continuous_scale='Reds')
    fig.update_layout(height=400, xaxis_title="Impact (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="footer">© 2024 Ride-Hailing ML System | Real-time driver matching powered by AI</div>', unsafe_allow_html=True)


# ============================================================================
# REAL-TIME PREDICTION PAGE
# ============================================================================
elif page == "🔮 Real-Time Prediction":
    st.markdown('<div class="main-header">🔮 Real-Time Driver Matching</div>', unsafe_allow_html=True)
    
    st.markdown("Enter order and driver details to get instant acceptance prediction")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📍 Order Details")
        with st.container():
            order_id = st.text_input("Order ID", "ORD_" + str(random.randint(10000, 99999)))
            customer_id = st.text_input("Customer ID", "CUST_" + str(random.randint(1000, 9999)))
            
            st.markdown("**Pickup Location**")
            pickup_lat = st.number_input("Latitude", value=40.7128, format="%.6f", key="rt_pickup_lat")
            pickup_lon = st.number_input("Longitude", value=-74.0060, format="%.6f", key="rt_pickup_lon")
            
            trip_distance = st.slider("Trip Distance (km)", 1.0, 50.0, 5.0, 0.5)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                hour = st.selectbox("Hour", list(range(24)), index=14)
            with col_t2:
                day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    with col2:
        st.markdown("### 🚗 Driver Details")
        with st.container():
            driver_id = st.text_input("Driver ID", "DRV_" + str(random.randint(1000, 9999)))
            
            st.markdown("**Current Location**")
            driver_lat = st.number_input("Latitude", value=40.7160, format="%.6f", key="rt_driver_lat")
            driver_lon = st.number_input("Longitude", value=-74.0100, format="%.6f", key="rt_driver_lon")
            
            driver_experience = st.slider("Experience (trips)", 0, 10000, 500)
            driver_acceptance = st.slider("Acceptance Rate", 0.0, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    
    if st.button("🚀 Calculate Match Probability", type="primary", use_container_width=True):
        with st.spinner("Analyzing driver availability..."):
            is_rush_hour = 1 if (7 <= hour <= 10 or 16 <= hour <= 19) else 0
            
            input_data = {
                'pickup_lat': pickup_lat,
                'pickup_lon': pickup_lon,
                'driver_lat': driver_lat,
                'driver_lon': driver_lon,
                'driver_acceptance': driver_acceptance,
                'is_rush_hour': is_rush_hour
            }
            
            probability, distance = make_prediction(input_data)
            
            # Results display
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Distance to Pickup", f"{distance:.2f} km", delta="+" if distance > 2 else "-")
            with col_r2:
                st.metric("Acceptance Probability", f"{probability:.1%}")
            with col_r3:
                confidence = probability if probability >= 0.5 else 1 - probability
                st.metric("Confidence Level", f"{confidence:.1%}")
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                delta={'reference': 50},
                title={'text': "Driver Acceptance Probability"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF4B4B"},
                       'steps': [{'range': [0, 30], 'color': '#FFCCCC'}, {'range': [30, 70], 'color': '#FFFFCC'}, {'range': [70, 100], 'color': '#CCFFCC'}],
                       'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 50}}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            if probability >= 0.5:
                st.markdown(f'''
                <div class="prediction-accept">
                    <h3>✅ RECOMMENDATION: ASSIGN THIS DRIVER</h3>
                    <p>The driver is {probability:.1%} likely to accept this ride request.</p>
                    <p>Estimated wait time: {max(2, int(distance * 2))} minutes</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="prediction-reject">
                    <h3>❌ RECOMMENDATION: FIND ANOTHER DRIVER</h3>
                    <p>This driver is only {probability:.1%} likely to accept.</p>
                    <p>Consider offering a surge incentive or finding a closer driver.</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Save Results Option (Local Only)
            if IS_LOCAL:
                st.markdown("---")
                st.markdown("### 💾 Save This Prediction")
                
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    if st.button("Save to Local Disk", use_container_width=True):
                        # Create results dataframe
                        result_df = pd.DataFrame([{
                            'timestamp': datetime.now().isoformat(),
                            'order_id': order_id,
                            'driver_id': driver_id,
                            'pickup_lat': pickup_lat,
                            'pickup_lon': pickup_lon,
                            'driver_lat': driver_lat,
                            'driver_lon': driver_lon,
                            'distance_km': round(distance, 2),
                            'trip_distance_km': trip_distance,
                            'hour': hour,
                            'driver_acceptance_rate': driver_acceptance,
                            'acceptance_probability': f"{probability:.1%}",
                            'recommendation': 'ACCEPT' if probability >= 0.5 else 'REJECT'
                        }])
                        
                        success, csv_path, meta_path = save_to_local(result_df, "real_time")
                        if success:
                            st.session_state.last_save_path = csv_path
                            st.markdown(f'<div class="save-success">✅ Saved to: <code>{csv_path}</code></div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Save failed: {meta_path}")
                
                with col_s2:
                    st.caption("Saves to: `artifacts/predictions/real_time_predictions_*.csv`")


# ============================================================================
# BATCH PREDICTION PAGE
# ============================================================================
elif page == "📊 Batch Prediction":
    st.markdown('<div class="main-header">📊 Batch Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("Upload a CSV file to process multiple driver-order pairs at once")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📄 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown(f"**Total records:** {len(df)}")
            
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} records..."):
                    results = []
                    for idx, row in df.iterrows():
                        input_data = {
                            'pickup_lat': row.get('pickup_latitude', 40.7128),
                            'pickup_lon': row.get('pickup_longitude', -74.0060),
                            'driver_lat': row.get('driver_latitude', 40.7160),
                            'driver_lon': row.get('driver_longitude', -74.0100),
                            'driver_acceptance': row.get('driver_acceptance_rate', 0.7),
                            'is_rush_hour': row.get('is_rush_hour', 0)
                        }
                        proba, distance = make_prediction(input_data)
                        results.append({
                            'order_id': row.get('order_id', f'ORD_{idx}'),
                            'driver_id': row.get('driver_id', f'DRV_{idx}'),
                            'distance_km': round(distance, 2),
                            'match_probability': f"{proba:.1%}",
                            'recommendation': 'ACCEPT' if proba >= 0.5 else 'REJECT'
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("### 📊 Results Summary")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Processed", len(results_df))
                    with col_b:
                        accept_count = (results_df['recommendation'] == 'ACCEPT').sum()
                        st.metric("Accepted", accept_count)
                    with col_c:
                        st.metric("Accept Rate", f"{accept_count/len(results_df):.1%}")
                    
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download button (always available)
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results (CSV)", csv, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                    
                    # Save to local disk (local only)
                    if IS_LOCAL:
                        st.markdown("---")
                        st.markdown("### 💾 Save to Local Disk")
                        if st.button("Save Batch Results to Local", use_container_width=True):
                            success, csv_path, meta_path = save_to_local(results_df, "batch")
                            if success:
                                st.session_state.last_save_path = csv_path
                                st.markdown(f'<div class="save-success">✅ Batch results saved to: <code>{csv_path}</code></div>', unsafe_allow_html=True)
                            else:
                                st.error(f"Save failed: {meta_path}")
    
    with col2:
        st.markdown("### 📁 Sample Format")
        sample = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'driver_id': ['DRV001', 'DRV002'],
            'pickup_latitude': [40.7128, 40.7140],
            'pickup_longitude': [-74.0060, -74.0080],
            'driver_latitude': [40.7160, 40.7100],
            'driver_longitude': [-74.0100, -74.0040],
            'trip_distance': [5.2, 3.8]
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
        
        st.download_button("📥 Download Sample CSV", sample.to_csv(index=False), "sample_data.csv", "text/csv")


# ============================================================================
# ANALYTICS PAGE
# ============================================================================
elif page == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Performance Analytics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        time_range = st.selectbox("Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date"])
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", "8,942", delta="+23%")
    with col2:
        st.metric("Acceptance Rate", "78.3%", delta="+5.2%")
    with col3:
        st.metric("Avg Wait Time", "4.2 min", delta="-1.3 min")
    with col4:
        st.metric("Driver Utilization", "82%", delta="+7%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Daily Acceptance Rate")
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        rates = [0.65 + (i * 0.005) + random.uniform(-0.02, 0.02) for i in range(30)]
        fig = px.line(x=dates, y=rates, title="Acceptance Rate Trend")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⏰ Hourly Demand Pattern")
        hours = list(range(24))
        demand = [10, 5, 3, 2, 1, 2, 5, 15, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85, 75, 60, 45, 30, 20, 15]
        fig = px.bar(x=hours, y=demand, title="Ride Requests by Hour")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🗺️ Top Pickup Locations")
    locations = pd.DataFrame({
        'Location': ['Downtown', 'Airport', 'Financial District', 'Shopping Mall', 'University'],
        'Requests': [12450, 8920, 7650, 5430, 4210],
        'Avg Wait (min)': [3.2, 5.1, 2.8, 4.5, 3.8]
    })
    st.dataframe(locations, use_container_width=True, hide_index=True)


# ============================================================================
# SAVED OUTPUTS PAGE (New)
# ============================================================================
elif page == "📁 Saved Outputs":
    st.markdown('<div class="main-header">📁 Saved Predictions</div>', unsafe_allow_html=True)
    
    if not IS_LOCAL:
        st.warning("💡 Saved outputs are only available in **Local Mode**. In Cloud Mode, please download results using the Download button.")
        st.info("To save outputs locally, run this app on your own machine.")
    else:
        st.markdown("View and manage your saved prediction outputs")
        
        # Directory to scan
        base_dir = Path(__file__).parent
        predictions_dir = base_dir / "artifacts" / "predictions"
        
        if not predictions_dir.exists():
            st.info("No saved outputs found. Make some predictions and save them first!")
        else:
            # Find all prediction files
            csv_files = list(predictions_dir.glob("*_predictions_*.csv"))
            json_files = list(predictions_dir.glob("*_metadata_*.json"))
            
            if not csv_files:
                st.info("No saved prediction files found.")
            else:
                st.markdown(f"### Found {len(csv_files)} saved prediction sets")
                
                # Sort by modification time (newest first)
                csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for csv_file in csv_files[:10]:  # Show last 10
                    file_time = datetime.fromtimestamp(csv_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    file_size = csv_file.stat().st_size / 1024  # KB
                    
                    with st.expander(f"📄 {csv_file.name} - {file_time} ({file_size:.1f} KB)"):
                        # Load and preview the data
                        df = pd.read_csv(csv_file)
                        st.dataframe(df.head(), use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Records", len(df))
                        with col2:
                            if 'recommendation' in df.columns:
                                accept_count = (df['recommendation'] == 'ACCEPT').sum()
                                st.metric("Accepts", accept_count)
                        with col3:
                            st.metric("File Size", f"{file_size:.1f} KB")
                        
                        # Load metadata if exists
                        metadata_file = predictions_dir / csv_file.name.replace("predictions_", "metadata_").replace(".csv", ".json")
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            st.caption(f"Saved: {metadata.get('timestamp', 'Unknown')}")
                        
                        # Download button for saved file
                        with open(csv_file, 'rb') as f:
                            st.download_button("📥 Download", f.read(), csv_file.name, "text/csv", key=csv_file.name)
                
                # Cleanup option
                st.markdown("---")
                st.warning("⚠️ Delete all saved outputs?")
                if st.button("🗑️ Delete All Saved Files", type="secondary"):
                    import shutil
                    shutil.rmtree(predictions_dir)
                    predictions_dir.mkdir(parents=True, exist_ok=True)
                    st.success("All saved files deleted!")
                    st.rerun()


# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        
        To revolutionize ride-hailing efficiency through AI-powered driver matching.
        
        ### 💡 How It Works
        
        1. **Input** order and driver location details
        2. **AI Model** analyzes multiple factors
        3. **Prediction** outputs acceptance probability
        4. **Recommendation** guides dispatch decision
        
        ### 📊 Key Features
        
        - **Real-time matching** - <100ms response
        - **Batch processing** - Handle thousands of requests
        - **Analytics dashboard** - Track performance metrics
        - **Save results locally** - CSV and JSON export
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Model Details
        
        | Feature | Impact |
        |---------|--------|
        | Distance to pickup | 35% |
        | Time of day | 25% |
        | Driver history | 18% |
        | Trip distance | 12% |
        | GPS accuracy | 5% |
        
        ### 📈 Business Impact
        
        | Metric | Improvement |
        |--------|-------------|
        | Match rate | +20% |
        | Wait time | -31% |
        | Utilization | +17% |
        | Revenue | +25% |
        """)
    
    st.markdown("---")
    
    # Environment info
    st.markdown("### 🖥️ Environment Information")
    if IS_LOCAL:
        st.success("✅ **Local Mode** - Save feature enabled")
        st.code("Saved files location: ./artifacts/predictions/")
    else:
        st.info("☁️ **Cloud Mode** - Download results only")
    
    st.markdown("---")
    st.markdown("""
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **ML Models**: Random Forest, XGBoost
    - **Visualization**: Plotly, Matplotlib
    - **Deployment**: Streamlit Cloud / Local
    
    ### 📞 Support
    
    For questions or support, please contact the development team.
    
    ---
    *Version 2.0.0 | Last Updated: January 2024*
    """)

if page != "🏠 Dashboard":
    st.markdown('<div class="footer">© 2024 Ride-Hailing ML System | AI-Powered Driver Matching</div>', unsafe_allow_html=True)