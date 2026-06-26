import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. PAGE SETUP & HIGH-END VIEWPORT ARCHITECTURE
# ==============================================================================
st.set_page_config(
    layout="wide", 
    page_title="SkyCity Operational Intelligence Matrix", 
    page_icon="🔮"
)

# Custom Ultra-Premium CSS Injection (Glassmorphism + Neon Accents + Animations)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Core Layout Styles */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #12192c 0%, #080b11 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    /* Premium Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #05070b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.15) 0%, rgba(0, 176, 255, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Glassmorphic Panel/Card Architecture */
    .glass-panel {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    .glass-panel:hover {
        border-color: rgba(0, 230, 118, 0.4);
        box-shadow: 0 12px 40px 0 rgba(0, 230, 118, 0.12);
        transform: translateY(-3px);
    }
    
    /* Custom High-End KPI Layouts */
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
    }
    
    /* Operational Recommendation Custom Alert Cards */
    .rec-card {
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 4px solid;
    }
    .rec-success { background: rgba(0, 230, 118, 0.06); border-left-color: #00e676; color: #a7f3d0; }
    .rec-warning { background: rgba(255, 167, 38, 0.06); border-left-color: #ffa726; color: #ffedd5; }
    .rec-error { background: rgba(239, 83, 80, 0.06); border-left-color: #ef5350; color: #fee2e2; }
    
    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 12px 28px;
        color: #94a3b8;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #00e676 !important;
        border-color: rgba(0, 230, 118, 0.3) rgba(0, 230, 118, 0.3) transparent rgba(0, 230, 118, 0.3) !important;
    }
    
    /* Animation Framework */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATASET & MACHINE LEARNING MODEL INGESTION
# ==============================================================================
@st.cache_resource
def load_analytics_pipeline():
    model = joblib.load('best_profit_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    raw_data = pd.read_csv("dataset/SkyCity_Restaurants_Cleaned.csv")
    return model, preprocessor, raw_data

try:
    model, preprocessor, df = load_analytics_pipeline()
except Exception as e:
    st.error("Pipeline assets missing. Execute 'model_pipeline.py' inside your core workspace framework first.")
    st.stop()

# ==============================================================================
# 3. INTERACTIVE PROFESSIONAL SIDEBAR
# ==============================================================================
st.sidebar.markdown("<h2 style='color:#ffffff; font-size:1.4rem; margin-bottom:5px; font-weight:800;'>🔮 Control Matrix</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#64748b; font-size:0.85rem; margin-bottom:20px;'>Adjust operational variables in real-time</p>", unsafe_allow_html=True)

restaurant_list = sorted(df['RestaurantName'].unique())
selected_restaurant = st.sidebar.selectbox("🏪 Select Target Asset", restaurant_list)
base_row = df[df['RestaurantName'] == selected_restaurant].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color:#ffffff; font-size:1rem; margin-bottom:10px;'>📊 Dynamic Channel Allocation</h4>", unsafe_allow_html=True)

instore_share = st.sidebar.slider("In-Store Mix Share", 0.0, 1.0, float(base_row['InStoreShare']), 0.01)
ue_share = st.sidebar.slider("Uber Eats Mix Share", 0.0, 1.0, float(base_row['UE_share']), 0.01)
dd_share = st.sidebar.slider("DoorDash Mix Share", 0.0, 1.0, float(base_row['DD_share']), 0.01)
sd_share = st.sidebar.slider("Self-Delivery Mix Share", 0.0, 1.0, float(base_row['SD_share']), 0.01)

# Dynamic Validation System
total_share = instore_share + ue_share + dd_share + sd_share
if not np.isclose(total_share, 1.0, atol=0.01):
    st.sidebar.error(f"⚠️ Vector Balance Failure: Mix equals {total_share*100:.1f}%. Rebalance sliders to exactly 100%.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color:#ffffff; font-size:1rem; margin-bottom:10px;'>💵 Cost Parameter Deviations</h4>", unsafe_allow_html=True)
comm_rate = st.sidebar.slider("Aggregator Commission Base", 0.15, 0.40, float(base_row['CommissionRate']), 0.01)
del_cost = st.sidebar.slider("Fulfillment Cost / Order ($)", 0.50, 6.00, float(base_row['DeliveryCostPerOrder']), 0.10)

# ==============================================================================
# 4. COMPILATION & ML PREDICTION ENGINE
# ==============================================================================
simulated_data = pd.DataFrame([{
    'CuisineType': base_row['CuisineType'],
    'Segment': base_row['Segment'],
    'Subregion': base_row['Subregion'],
    'GrowthFactor': base_row['GrowthFactor'],
    'AOV': base_row['AOV'],
    'MonthlyOrders': base_row['MonthlyOrders'],
    'InStoreShare': instore_share,
    'UE_share': ue_share,
    'DD_share': dd_share,
    'SD_share': sd_share,
    'CommissionRate': comm_rate,
    'DeliveryRadiusKM': base_row['DeliveryRadiusKM'],
    'DeliveryCostPerOrder': del_cost,
    'UE_Cost_Interaction': comm_rate * ue_share,
    'DD_Cost_Interaction': comm_rate * dd_share,
    'SD_Cost_Interaction': del_cost * sd_share
}])

processed_input = preprocessor.transform(simulated_data)
predicted_profit = model.predict(processed_input)[0]

actual_profit = (base_row['InStoreNetProfit'] + base_row['UberEatsNetProfit'] + 
                 base_row['DoorDashNetProfit'] + base_row['SelfDeliveryNetProfit'])

profit_delta = predicted_profit - actual_profit
uplift_pct = (profit_delta / abs(actual_profit)) * 100

# ==============================================================================
# 5. WORKSPACE INTERFACE: PREMIUM HERO BANNER
# ==============================================================================
st.markdown(f"""
    <div class="hero-banner">
        <h1 style="margin:0; font-size:2.4rem; font-weight:800; color:#fff; letter-spacing:-1px;">🔮 SkyCity Intelligence Matrix</h1>
        <p style="margin:5px 0 0 0; color:#a2b6cf; font-size:1.1rem;">Prescriptive Channel Optimization Map for <span style="color:#00e676; font-weight:600;">{selected_restaurant}</span></p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 6. HIGH-END KPI GLASS CARDS
# ==============================================================================
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.markdown(f"""
        <div class="glass-panel">
            <div class="kpi-title">Historical Baseline Profit</div>
            <div class="kpi-value" style="color:#ffffff;">${actual_profit:,.2f}</div>
            <div class="kpi-delta" style="color:#64748b;">Audited Historic Period</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    delta_color = "#00e676" if profit_delta >= 0 else "#ef5350"
    arrow = "▲" if profit_delta >= 0 else "▼"
    st.markdown(f"""
        <div class="glass-panel">
            <div class="kpi-title">Simulated Scenario Profit</div>
            <div class="kpi-value" style="color:#00b0ff;">${predicted_profit:,.2f}</div>
            <div class="kpi-delta" style="color:{delta_color};">{arrow} ${abs(profit_delta):,.2f} Variance</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    badge_color = "#00e676" if uplift_pct >= 0 else "#ef5350"
    st.markdown(f"""
        <div class="glass-panel">
            <div class="kpi-title">Strategy Optimization Lift</div>
            <div class="kpi-value" style="color:{badge_color};">{uplift_pct:+.2f}%</div>
            <div class="kpi-delta" style="color:#94a3b8;">Expected Margin Shift</div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 7. EXECUTIVE WORKSPACE TABS
# ==============================================================================
tab_viz, tab_ai, tab_profile = st.tabs(["📊 Executive Visualization Matrix", "🧠 AI Insights & Prescriptive Decisions", "🏪 Asset Operational Profile"])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE VISUALIZATION MATRIX
# ------------------------------------------------------------------------------
with tab_viz:
    # --- ROW 1: EQUAL SYMMETRIC SPLIT ---
    row1_g1, row1_g2 = st.columns([1, 1])
    
    with row1_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_profit,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Simulated Margin Target ($)", 'font': {'size': 16, 'color': '#ffffff'}},
            gauge = {
                'axis': {'range': [df['InStoreNetProfit'].min(), df['TotalNetProfit'].max()], 'tickcolor': "#94a3b8"},
                'bar': {'color': "#00b0ff"},
                'bgcolor': "rgba(255,255,255,0.03)",
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [df['InStoreNetProfit'].min(), 0], 'color': 'rgba(239, 83, 80, 0.1)'},
                    {'range': [0, df['TotalNetProfit'].max()], 'color': 'rgba(0, 230, 118, 0.05)'}
                ],
                'threshold': {'line': {'color': "#00e676", 'width': 4}, 'thickness': 0.75, 'value': actual_profit}
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=20, l=30, r=30), height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with row1_g2:
        labels = ['In-Store Mix', 'Uber Eats Mix', 'DoorDash Mix', 'Self-Delivery Mix']
        values = [instore_share, ue_share, dd_share, sd_share]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.45,
            marker=dict(colors=['#00e676', '#00b0ff', '#ff3d00', '#ffea00']),
            textinfo='percent+label', hoverinfo='label+percent',
            textfont_size=11
        )])
        fig_donut.update_layout(
            title=dict(text="Simulated Cross-Channel Volume Share Mix", font=dict(color='#ffffff', size=16)),
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=20, r=20), height=400, showlegend=False
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # --- ROW 2: THE MIDDLE MATRIX ---
    row2_g1, row2_g2 = st.columns([1, 1])
    
    with row2_g1:
        try:
            rates = np.linspace(0.15, 0.40, 15)
            curve_profits = []
            for r in rates:
                temp_data = simulated_data.copy()
                temp_data['CommissionRate'] = r
                temp_data['UE_Cost_Interaction'] = r * ue_share
                temp_data['DD_Cost_Interaction'] = r * dd_share
                curve_profits.append(model.predict(preprocessor.transform(temp_data))[0])
                
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=rates * 100, y=curve_profits, mode='lines',
                line=dict(color='#00e676', width=4, shape='spline'),
                fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.03)',
                hovertemplate="Fee Tier: %{x:.1f}%<br>Profit: $%{y:,.2f}<extra></extra>"
            ))
            fig_curve.update_layout(
                title=dict(text="Commission Elasticity Vector (Net Return vs Fees)", font=dict(color='#ffffff', size=16)),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Commission Rate (%)"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Predicted Return ($)"),
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=60, b=40), height=400
            )
            st.plotly_chart(fig_curve, use_container_width=True)
        except Exception as chart_err:
            st.error(f"Failed to compile elasticity metrics: {chart_err}")

    with row2_g2:
        try:
            # Compute foundational shifts relative to baseline metrics
            instore_delta = (instore_share - base_row['InStoreShare']) * base_row['MonthlyOrders'] * base_row['AOV'] * (1 - base_row['COGSRate'] - base_row['OPEXRate'])
            
            hist_ue = max(0.01, base_row['UE_share'])
            hist_dd = max(0.01, base_row['DD_share'])
            
            ue_delta = base_row['UberEatsNetProfit'] * ((ue_share - base_row['UE_share']) / hist_ue)
            dd_delta = base_row['DoorDashNetProfit'] * ((dd_share - base_row['DD_share']) / hist_dd)
            sd_delta = (base_row['SelfDeliveryNetProfit'] if base_row['SD_share'] > 0 else 0) * (sd_share - base_row['SD_share'])

            y_values = [
                actual_profit, 
                instore_delta if instore_delta != 0 else 0.1, 
                ue_delta if ue_delta != 0 else -0.1, 
                dd_delta if dd_delta != 0 else -0.1, 
                sd_delta if sd_delta != 0 else -0.1, 
                predicted_profit
            ]

            fig_water = go.Figure(go.Waterfall(
                name = "Profit Extraction", orientation = "v",
                measure = ["absolute", "relative", "relative", "relative", "relative", "total"],
                x = ["Historic Baseline", "In-Store Shift", "Uber Eats Shift", "DoorDash Shift", "Self-Deliv Shift", "Simulated State"],
                textposition = "outside",
                text = [f"${actual_profit:,.0f}", f"${instore_delta:,.0f}", f"${ue_delta:,.0f}", f"${dd_delta:,.0f}", f"${sd_delta:,.0f}", f"${predicted_profit:,.0f}"],
                y = y_values,
                connector = {"line":{"color":"rgba(255,255,255,0.2)"}},
                decreasing = {"marker":{"color":"#ef5350"}},
                increasing = {"marker":{"color":"#00e676"}},
                totals = {"marker":{"color":"#00b0ff"}}
            ))
            fig_water.update_layout(
                title = dict(text="Strategic Margin Value Waterfall Map (Delta Tracking)", font=dict(color='#ffffff', size=16)),
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=60, b=40), height=400, showlegend=False
            )
            st.plotly_chart(fig_water, use_container_width=True)
        except Exception as chart_err:
            st.error(f"Failed to map waterfall vectors: {chart_err}")
            
    # --- ROW 3: REVENUE VARIANCE TREND MAP ---
    subregion_avg = df[df['Subregion'] == base_row['Subregion']].groupby('Segment')['TotalNetProfit'].mean().reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=subregion_avg['Segment'], y=subregion_avg['TotalNetProfit'],
        name='Subregional Baseline Mean', marker_color='rgba(255,255,255,0.15)', width=0.4
    ))
    fig_trend.add_trace(go.Scatter(
        x=[base_row['Segment']], y=[predicted_profit],
        mode='markers', name='This Asset Simulation Point',
        marker=dict(color='#00e676', size=16, line=dict(color='#ffffff', width=2))
    ))
    fig_trend.update_layout(
        title=dict(text=f"Market Segment Benchmark Performance Map inside {base_row['Subregion']}", font=dict(color='#ffffff', size=16)),
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=60, b=40), height=360
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: AI INSIGHTS & PRESCRIPTIVE DECISIONS
# ------------------------------------------------------------------------------
with tab_ai:
    ai_col1, ai_col2 = st.columns(2)
    
    with ai_col1:
        st.markdown("### 🧠 Autonomous AI Insights Layer")
        st.markdown("<div class='glass-panel' style='min-height:320px;'>", unsafe_allow_html=True)
        st.write(f"**Contextual Framework Ingestion Engine Log for {selected_restaurant}:**")
        
        if predicted_profit > actual_profit:
            st.write(f"🔮 **Optimization Signal Identified:** The user-defined scenario introduces a structural optimization path shifting volume out of low-yielding streams. This reconfiguration captures an additional **${profit_delta:,.2f}** in monthly net returns.")
        else:
            st.write("🔮 **Margin Dilution Risk Detected:** The active operational configuration decreases overall efficiency. Current channel migration patterns indicate value erosion rather than growth.")
            
        st.write(f"📈 **Volume Elasticity Baseline:** Based on an Average Order Value of **${base_row['AOV']:.2f}**, a 5% systemic increase in baseline walk-in transactions (`InStoreShare`) provides a higher structural protection margin against third-party platforms than altering logistics models.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with ai_col2:
        st.markdown("### ⚠️ Dynamic Exposure Risk Analysis")
        st.markdown("<div class='glass-panel' style='min-height:320px;'>", unsafe_allow_html=True)
        
        if ue_share + dd_share > 0.55:
            st.markdown("""
                <div class='rec-card rec-error'>
                    <strong>CRITICAL EXPOSURE: Aggregator Dependency Fracture</strong><br>
                    Over 55% of your volume runs through aggregators. At current commission settings, platform fees erode margins significantly. Immediate conversion to direct consumer order loops is advised.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='rec-card rec-success'>
                    <strong>EXPOSURE EXCELLENCE: Intermediary Protection</strong><br>
                    Aggregator dependency indexes remain inside safe operational parameters. Your configuration successfully protects core margins from external fee pressure.
                </div>
            """, unsafe_allow_html=True)
            
        if sd_share > 0.30 and del_cost > 3.80:
            st.markdown(f"""
                <div class='rec-card rec-warning'>
                    <strong>VARIABLE WARNING: Logistics Cost Inefficiency</strong><br>
                    Internal delivery cost scales aggressively (${del_cost:,.2f}/order). Logistics infrastructure is creating structural margin leakage. Consider outsourcing or introducing a radius cap.
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # EXECUTIVE RECOMMENDATIONS FRAMEWORK PANEL
    st.markdown("### 📋 Prescriptive Executive Action Plan")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown(f"""
        1. **Negotiation Target Framework:** Utilize the generated elasticity matrix to establish a benchmark for commission caps. If platform providers exceed a **{comm_rate*100:.1f}%** commission threshold, redirect fulfillment towards alternative channels.
        2. **Logistics Optimization Vector:** Cap maximum outreach parameters to contain delivery costs to under **${del_cost:.2f}** per order. This safeguards the profitability of your self-managed logistics network.
        3. **Dynamic Resource Reallocation:** Based on simulated performance patterns for the **{base_row['Segment']}** business format, allocate capital expenditures toward scaling local on-premise order capture.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: ASSET OPERATIONAL PROFILE
# ------------------------------------------------------------------------------
with tab_profile:
    st.markdown("### 🏪 Asset Analytical Configuration Summary")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    
    profile_matrix = pd.DataFrame({
        'Operational Core Parameter Attribute': [
            'System Segment Model Classification', 
            'Subregional Asset Territory Assignment', 
            'Audited Historical Average Order Value (AOV)', 
            'Baseline Volume (Orders/Month)', 
            'Configured Structural Radius Parameters'
        ],
        'Active Node Profile Value': [
            str(base_row['Segment']), 
            str(base_row['Subregion']), 
            f"${base_row['AOV']:.2f}", 
            f"{int(base_row['MonthlyOrders']):,}", 
            f"{base_row['DeliveryRadiusKM']} Kilometers"
        ]
    })
    st.table(profile_matrix)
    st.markdown("</div>", unsafe_allow_html=True)