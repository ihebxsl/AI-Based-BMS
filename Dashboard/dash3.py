import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import uuid
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.layers import SpatialDropout1D
from plotly.subplots import make_subplots

# Custom SpatialDropout1D class to fix deserialization error
class SpatialDropout1DCustom(SpatialDropout1D):
    @classmethod
    def from_config(cls, config):
        for key in ['trainable', 'noise_shape', 'seed']:
            config.pop(key, None)
        return super().from_config(config)

# ── Global Styling ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #0b0c10;
    }
    .main {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #66fcf1;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric {
        background-color: #1f2833;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0 10px #45a29e;
        color: #ffffff;
    }
    .small-map > div {
        height: 200px !important;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 0 10px #45a29e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── 1. Title & Upload ──────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>🚘 Car Dashboard Simulator</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload Driving Data (JSON or CSV)", type=["json", "csv"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", delimiter=";")

    

    # Load TCN model and scalers
    tcn_model = load_model('tcn_model1.h5', custom_objects={'SpatialDropout1D': SpatialDropout1DCustom}, compile=False)
    feature_scaler = joblib.load("feature_scaler1.pkl")
    target_scaler = joblib.load("target_scaler1.pkl")

    # Define GPS route boundaries
    LAT_START, LAT_END = 52.50, 52.55
    LON_START, LON_END = 13.35, 13.40

    df.columns = df.columns.str.strip()
    df["Time [s]"] = pd.to_numeric(df["Time [s]"], errors="coerce")
    df = df.dropna(subset=["Time [s]"])

    t_min, t_max = df["Time [s]"].min(), df["Time [s]"].max()
    t_span = t_max - t_min if t_max != t_min else 1

    df["Latitude"] = LAT_START + (df["Time [s]"] - t_min) / t_span * (LAT_END - LAT_START)
    df["Longitude"] = LON_START + (df["Time [s]"] - t_min) / t_span * (LON_END - LON_START)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").ffill()

    # Feature columns expected by model (47 features)
    original_feature_cols = [
        'Time [s]', 'Velocity [km/h]', 'Elevation [m]', 'Throttle [%]', 'Motor Torque [Nm]', 'Longitudinal Acceleration [m/s^2]',
        'Regenerative Braking Signal ', 'Battery Voltage [V]', 'Battery Current [A]', 'Battery Temperature [°C]',
        'max. Battery Temperature [°C]', 'displayed SoC [%]', 'min. SoC [%]', 'max. SoC [%]',
        'Heating Power CAN [kW]', 'Heating Power LIN [W]', 'Requested Heating Power [W]',
        'AirCon Power [kW]', 'Heater Signal', 'Heater Voltage [V]', 'Heater Current [A]',
        'Ambient Temperature [°C]', 'Ambient Temperature Sensor [°C]', 'Coolant Temperature Heatercore [°C]',
        'Requested Coolant Temperature [°C]', 'Coolant Temperature Inlet [°C]', 'Coolant Volume Flow +500 [l/h]',
        'Heat Exchanger Temperature [°C]', 'Cabin Temperature Sensor [°C]', 'Temperature Coolant Heater Inlet [°C]',
        'Temperature Coolant Heater Outlet [°C]', 'Temperature Heat Exchanger Outlet [°C]',
        'Temperature Defrost lateral left [°C]', 'Temperature Defrost lateral right [°C]',
        'Temperature Defrost central [°C]', 'Temperature Defrost central left [°C]',
        'Temperature Defrost central right [°C]', 'Temperature Footweel Driver [°C]',
        'Temperature Footweel Co-Driver [°C]', 'Temperature Feetvent Co-Driver [°C]',
        'Temperature Feetvent Driver [°C]', 'Temperature Head Co-Driver [°C]', 'Temperature Head Driver [°C]',
        'Temperature Vent right [°C]', 'Temperature Vent central right [°C]', 'Temperature Vent central left [°C]',
        'Temperature Vent right [°C]'
    ]

    original_feature_cols = [col.strip() for col in original_feature_cols]

    def predict_soc_tcn(df, current_index, model, future_offset_seconds=30):
        lookback = 10  # time steps used as input
        future_offset = int(future_offset_seconds * 10)  # 10 rows per second
        
        prediction_index = current_index + future_offset

        if current_index < lookback or prediction_index >= len(df):
            return None

        # Extract input window for current time
        input_window = df.iloc[current_index - lookback:current_index]

        # Fill missing columns if any
        for col in original_feature_cols:
            if col not in input_window.columns:
                input_window[col] = 0

        input_window = input_window[original_feature_cols]

        # Scale and predict
        scaled_features = feature_scaler.transform(input_window)
        input_seq = scaled_features.reshape(1, lookback, len(original_feature_cols))
        pred_scaled = model.predict(input_seq)[0, 0]
        pred_soc = target_scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
        return pred_soc

    def create_gauge(value, title, min_val, max_val, unit):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': f"<b>{title}</b><br><span style='font-size:0.9em;color:#aaaaaa'>{unit}</span>",
                'font': {'size': 12}
            },
            gauge={
                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#66fcf1"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#45a29e",
                'steps': [
                    {'range': [min_val, min_val + (max_val - min_val) * 0.4], 'color': '#1f2833'},
                    {'range': [min_val + (max_val - min_val) * 0.4, min_val + (max_val - min_val) * 0.8], 'color': '#1a222d'},
                    {'range': [min_val + (max_val - min_val) * 0.8, max_val], 'color': '#151a23'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9}
            }
        ))
        fig.update_layout(
        height=240,  # increased from 200
        margin=dict(t=60, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Arial', 'size': 12}
        )
        return fig

    st.markdown("## ⏱️ Real-Time Driving Trip")
    # ── NEW: user-selectable prediction horizon ──
    prediction_horizon = st.selectbox(
        "🔮 Prediction Horizon",
        options=[30, 60, 120, 300],  # 30s, 1min, 2min, 5min
        format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x} sec",
        index=0  # default: 30s
    )
    min_time, max_time = int(t_min), int(t_max)
    time_placeholder = st.empty()
    metrics_placeholder = st.empty()
    map_placeholder = st.empty()
    analytics_placeholder = st.empty()  # placeholder for real-time plots
    soc_comparison_placeholder = st.empty()  # NEW: placeholder for SoC comparison plot

    autoplay = st.checkbox("▶️ Auto-play Trip", value=False)

    def show_frame(current_time: int):
        current_row = df.loc[df["Time [s]"] == current_time]
        if current_row.empty:
            return
        row = current_row.iloc[0]

        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_gauge(row['Velocity [km/h]'], "Speed", 0, 200, "km/h"), use_container_width=True, key=f"speed_{uuid.uuid4()}")
            with col2:
                st.plotly_chart(create_gauge(row.get('SoC [%]', 0), "SoC", 0, 100, "%"), use_container_width=True, key=f"soc_{uuid.uuid4()}")
            with col3:
                current_index = df.index[df["Time [s]"] == current_time]
                if not current_index.empty:
                    predicted_soc = predict_soc_tcn(df, current_index[0], tcn_model, future_offset_seconds=prediction_horizon)
                else:
                    predicted_soc = None

                horizon_label = f"{prediction_horizon//60} min" if prediction_horizon >= 60 else f"{prediction_horizon} sec"
                st.plotly_chart(
                    create_gauge(
                        predicted_soc if predicted_soc is not None else 0,
                        f"Predicted SoC (TCN) after {horizon_label}",
                        0, 100, "%"
                    ),
                    use_container_width=True,
                    key=f"pred_soc_{uuid.uuid4()}"
                )
                

            col4, col5, col6 = st.columns(3)
            with col4:
                power = row['Battery Voltage [V]'] * row['Battery Current [A]'] / 1000
                st.plotly_chart(create_gauge(power, "Power", 0, 300, "kW"), use_container_width=True, key=f"power_{uuid.uuid4()}")
            with col5:
                st.plotly_chart(create_gauge(row['AirCon Power [kW]'], "AirCon", 0, 10, "kW"), use_container_width=True, key=f"aircon_{uuid.uuid4()}")
            with col6:
                st.plotly_chart(create_gauge(row['Battery Temperature [°C]'], "Batt Temp", -20, 60, "°C"), use_container_width=True, key=f"batt_temp_{uuid.uuid4()}")

        with map_placeholder.container():
            st.markdown('<div class="small-map">', unsafe_allow_html=True)
            st.map(pd.DataFrame({"lat": [row["Latitude"]], "lon": [row["Longitude"]]}), zoom=12)
            st.markdown('</div>', unsafe_allow_html=True)

    def show_analytics_plots(current_time: int):
        if current_time < 10:
            analytics_placeholder.empty()
            return

        recent_df = df[df["Time [s]"] <= current_time].tail(100)  # last 100 points for smoother real-time effect

        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "Velocity [km/h]", "Battery Temp [°C]",
            "AirCon Power [kW]", "SoC [%]"
        ])

        fig.add_trace(go.Scatter(x=recent_df["Time [s]"], y=recent_df["Velocity [km/h]"],
                                 mode="lines", name="Speed", line=dict(color="#66fcf1")), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=recent_df["Time [s]"], y=recent_df["Battery Temperature [°C]"],
                                 mode="lines", name="Batt Temp", line=dict(color="#ff6b6b")), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=recent_df["Time [s]"], y=recent_df["AirCon Power [kW]"],
                                 mode="lines", name="AirCon", line=dict(color="#45a29e")), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=recent_df["Time [s]"], y=recent_df["SoC [%]"],
                                 mode="lines", name="SoC", line=dict(color="#f5b041")), row=2, col=2)

        fig.update_layout(
            height=600,
            margin=dict(t=30, b=10, l=10, r=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )

        for axis in fig['layout']:
            if axis.startswith('xaxis') or axis.startswith('yaxis'):
                fig['layout'][axis]['gridcolor'] = '#1f2833'

        analytics_placeholder.plotly_chart(fig, use_container_width=True)

    # NEW: Function to show real vs predicted SoC comparison
    def show_soc_comparison(current_time: int):
        if current_time < 10:
            soc_comparison_placeholder.empty()
            return

        # Get data up to current time
        history_df = df[df["Time [s]"] <= current_time]
        
        # Calculate predicted SoC for each point in history
        predicted_socs = []
        for i in range(len(history_df)):
            if i >= 10:  # need at least lookback samples
                pred = predict_soc_tcn(history_df, i, tcn_model, prediction_horizon)
                predicted_socs.append(pred)
            else:
                predicted_socs.append(None)
        
        history_df['Predicted_SoC'] = predicted_socs
        
        # Create comparison plot
        fig = go.Figure()
        
        # Add real SoC trace
        fig.add_trace(go.Scatter(
            x=history_df["Time [s]"],
            y=history_df["SoC [%]"],
            mode='lines',
            name='Actual SoC',
            line=dict(color='#66fcf1', width=2)
        ))
        
        # Add predicted SoC trace
        fig.add_trace(go.Scatter(
            x=history_df["Time [s]"],
            y=history_df['Predicted_SoC'],
            mode='lines',
            name=f'Predicted SoC ({prediction_horizon}s ahead)',
            line=dict(color='#f5b041', width=2, dash='dot')
        ))
        
        # Add current time marker
        fig.add_vline(
            x=current_time,
            line_width=2,
            line_dash="dash",
            line_color="white",
            annotation_text="Current Time",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title='Actual vs Predicted State of Charge',
            xaxis_title='Time [s]',
            yaxis_title='SoC [%]',
            height=400,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(gridcolor='#1f2833')
        fig.update_yaxes(gridcolor='#1f2833')
        
        soc_comparison_placeholder.plotly_chart(fig, use_container_width=True)

    if autoplay:
        for t in range(min_time, max_time + 1):
            time_placeholder.slider("Time [s]", min_time, max_time, value=t, step=1, disabled=True)
            show_frame(t)
            show_analytics_plots(t)
            show_soc_comparison(t)  # NEW: Show SoC comparison
            time.sleep(0.2)
            
    else:
        selected_time = time_placeholder.slider("Time [s]", min_time, max_time, value=min_time, step=1)
        show_frame(selected_time)
        show_analytics_plots(selected_time)
        show_soc_comparison(selected_time)  # NEW: Show SoC comparison

    # Manual Snapshot UI
    st.markdown("## 📊 Manual Snapshot")
    snapshot_time = st.slider("Select Time [s]", min_value=min_time, max_value=max_time, value=min_time, step=1)
    snapshot = df.loc[df["Time [s]"] == snapshot_time]

    if not snapshot.empty:
        r = snapshot.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric'><h3>🚗 Speed</h3><p>{r['Velocity [km/h]']:.1f} km/h</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'><h3>⚡ Power</h3><p>{r['Battery Voltage [V]'] * r['Battery Current [A]'] / 1000:.2f} kW</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric'><h3>🔋 SoC</h3><p>{r.get('SoC [%]', 0):.1f} %</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'><h3>🌬️ AirCon</h3><p>{r['AirCon Power [kW]']:.1f} kW</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric'><h3>🌡️ Batt Temp</h3><p>{r['Battery Temperature [°C]']:.1f} °C</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'><h3>🌡️ Cabin Temp</h3><p>{r['Cabin Temperature Sensor [°C]']:.1f} °C</p></div>", unsafe_allow_html=True)

        st.markdown('<div class="small-map">', unsafe_allow_html=True)
        st.map(pd.DataFrame({"lat": [r["Latitude"]], "lon": [r["Longitude"]]}), zoom=12)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 📈 Final Analytics Plots ────────────────────────────────────────────────
    st.markdown("## 📈 Analytics: Key Parameters Over Time")

    try:
        # Convert to numeric just to be sure
        df["Time [s]"] = pd.to_numeric(df["Time [s]"], errors="coerce")
        df["Velocity [km/h]"] = pd.to_numeric(df["Velocity [km/h]"], errors="coerce")
        df["Battery Temperature [°C]"] = pd.to_numeric(df["Battery Temperature [°C]"], errors="coerce")
        df["AirCon Power [kW]"] = pd.to_numeric(df["AirCon Power [kW]"], errors="coerce")
        df["SoC [%]"] = pd.to_numeric(df["SoC [%]"], errors="coerce")

        df_clean = df.dropna(subset=[
            "Time [s]", "Velocity [km/h]", "Battery Temperature [°C]",
            "AirCon Power [kW]", "SoC [%]"
        ])

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(df_clean, x="Time [s]", y="Velocity [km/h]", title="Velocity Over Time")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.line(df_clean, x="Time [s]", y="Battery Temperature [°C]", title="Battery Temperature Over Time")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.line(df_clean, x="Time [s]", y="AirCon Power [kW]", title="AirCon Power Over Time")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            # NEW: Enhanced SoC plot with predictions
            fig4 = go.Figure()
            
            # Add actual SoC
            fig4.add_trace(go.Scatter(
                x=df_clean["Time [s]"],
                y=df_clean["SoC [%]"],
                mode='lines',
                name='Actual SoC',
                line=dict(color='#66fcf1', width=2)
            ))
            
            # Add predicted SoC (calculate for full dataset)
            predicted_socs_full = []
            for i in range(len(df_clean)):
                if i >= 10:  # need at least lookback samples
                    pred = predict_soc_tcn(df_clean, i, tcn_model, prediction_horizon)
                    predicted_socs_full.append(pred)
                else:
                    predicted_socs_full.append(None)
            
            fig4.add_trace(go.Scatter(
                x=df_clean["Time [s]"],
                y=predicted_socs_full,
                mode='lines',
                name=f'Predicted SoC ({prediction_horizon}s ahead)',
                line=dict(color='#f5b041', width=2, dash='dot')
            ))
            
            fig4.update_layout(
                title='Actual vs Predicted State of Charge (Full Trip)',
                xaxis_title='Time [s]',
                yaxis_title='SoC [%]',
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig4.update_xaxes(gridcolor='#1f2833')
            fig4.update_yaxes(gridcolor='#1f2833')
            
            st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error generating analytics plots: {e}")

else:
    st.info("Please upload a driving data file to start the simulation.")