# streamlit_app.py
# Complete Streamlit application for Loan Approval Prediction
# To run: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/bagging_rf_model.pkl")
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'models/bagging_rf_model.pkl' not found. Please train the model first.")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("models/scaler.pkl")
        return scaler
    except FileNotFoundError:
        # Create a default scaler
        scaler = StandardScaler()
        sample_data = np.array([[2, 0, 0, 5000000, 15000000, 12, 700, 5000000, 3000000, 10000000, 5000000]])
        scaler.fit(sample_data)
        return scaler

def create_feature_dataframe(input_data):
    """Convert input data to proper dataframe format"""
    columns = [
        'no_of_dependents', 'education', 'self_employed', 'income_annum',
        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    return pd.DataFrame([input_data], columns=columns)

def predict_loan(model, scaler, input_data):
    """Make prediction using the model"""
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    return prediction[0], prediction_proba[0]

# Main App
def main():
    # Header
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("### Machine Learning Powered Decision Support System")
    st.markdown("---")

    # Load model
    model = load_model()
    scaler = load_scaler()

    if model is None:
        return

    # Sidebar for user input
    with st.sidebar:
        st.header("📋 Applicant Information")
        st.markdown("---")
        
        # Personal Information
        st.subheader("👤 Personal Details")
        
        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Number of family members dependent on the applicant"
        )
        
        education = st.selectbox(
            "Education Level",
            options=["Graduate", "Not Graduate"],
            help="Applicant's educational qualification"
        )
        
        self_employed = st.selectbox(
            "Self Employed",
            options=["No", "Yes"],
            help="Whether the applicant is self-employed"
        )
        
        # Financial Information
        st.subheader("💰 Financial Details")
        
        income_annum = st.number_input(
            "Annual Income (₹)",
            min_value=100000,
            max_value=10000000,
            value=5000000,
            step=100000,
            format="%d"
        )
        
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=100000,
            max_value=40000000,
            value=15000000,
            step=100000,
            format="%d"
        )
        
        loan_term = st.slider(
            "Loan Term (Years)",
            min_value=2,
            max_value=20,
            value=12,
            step=1
        )
        
        cibil_score = st.slider(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700,
            step=1,
            help="Credit score range: 300-900"
        )
        
        # Asset Information
        st.subheader("🏠 Asset Details")
        
        residential_assets_value = st.number_input(
            "Residential Assets Value (₹)",
            min_value=0,
            max_value=30000000,
            value=5000000,
            step=100000,
            format="%d"
        )
        
        commercial_assets_value = st.number_input(
            "Commercial Assets Value (₹)",
            min_value=0,
            max_value=20000000,
            value=3000000,
            step=100000,
            format="%d"
        )
        
        luxury_assets_value = st.number_input(
            "Luxury Assets Value (₹)",
            min_value=0,
            max_value=40000000,
            value=10000000,
            step=100000,
            format="%d"
        )
        
        bank_asset_value = st.number_input(
            "Bank Asset Value (₹)",
            min_value=0,
            max_value=15000000,
            value=5000000,
            step=100000,
            format="%d"
        )
    
    # Convert categorical variables
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 0 if self_employed == "No" else 1
    
    # Create input array
    input_data = [
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Application Summary")
        
        # Create two columns for better display
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Personal Information")
            st.write(f"**👥 Dependents:** {no_of_dependents}")
            st.write(f"**🎓 Education:** {education}")
            st.write(f"**💼 Self-Employed:** {self_employed}")
            st.write(f"**💳 CIBIL Score:** {cibil_score}")
        
        with col_b:
            st.subheader("Financial Information")
            st.write(f"**💰 Annual Income:** ₹{income_annum:,.0f}")
            st.write(f"**🏦 Loan Amount:** ₹{loan_amount:,.0f}")
            st.write(f"**📅 Loan Term:** {loan_term} years")
            total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
            st.write(f"**🏠 Total Assets:** ₹{total_assets:,.0f}")
        
        # Risk Assessment Section
        st.header("📈 Risk Assessment")
        
        # Calculate risk indicators
        loan_to_income_ratio = (loan_amount / income_annum) * 100
        asset_to_loan_ratio = (total_assets / loan_amount) * 100 if loan_amount > 0 else 0
        
        risk_metrics_col1, risk_metrics_col2, risk_metrics_col3 = st.columns(3)
        
        with risk_metrics_col1:
            st.metric(
                "Loan-to-Income Ratio",
                f"{loan_to_income_ratio:.1f}%",
                delta="High Risk" if loan_to_income_ratio > 300 else "Normal",
                delta_color="inverse"
            )
        
        with risk_metrics_col2:
            st.metric(
                "Asset-to-Loan Ratio",
                f"{asset_to_loan_ratio:.1f}%",
                delta="Good" if asset_to_loan_ratio > 100 else "Low",
                delta_color="normal"
            )
        
        with risk_metrics_col3:
            cibil_status = "Good" if cibil_score >= 700 else "Poor" if cibil_score < 500 else "Average"
            st.metric(
                "CIBIL Status",
                cibil_status,
                "✅ Good Credit" if cibil_score >= 700 else "⚠️ Needs Improvement",
                delta_color="normal" if cibil_score >= 700 else "inverse"
            )
        
        # Visual Analytics
        st.header("📊 Financial Profile Visualization")
        
        try:
            # Prepare data for radar chart with proper scaling
            radar_data = [
                min(income_annum / 1000000, 50),  # Income in millions
                min(loan_amount / 1000000, 40),   # Loan in millions
                cibil_score / 100,                 # CIBIL score scaled to 0-9
                min(total_assets / 1000000, 50),   # Assets in millions
                min(loan_to_income_ratio / 10, 50) # LTI ratio scaled
            ]
            
            categories = ['Income (M)', 'Loan (M)', 'CIBIL/100', 'Assets (M)', 'LTI Ratio/10']
            
            # Create radar chart using plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=radar_data,
                theta=categories,
                fill='toself',
                name='Applicant Profile',
                line=dict(color='#1E88E5', width=3),
                fillcolor='rgba(30, 136, 229, 0.2)',
                marker=dict(size=8, color='#1E88E5')
            ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 50],
                        color='#000000',
                        tickfont=dict(color='#000000'),
                        title=dict(text='Value', font=dict(color='#000000'))
                    ),
                    angularaxis=dict(
                        color='#000000',
                        tickfont=dict(color='#000000', size=12)
                    )
                ),
                height=500,
                showlegend=True,
                title=dict(
                    text="Financial Profile Radar Chart",
                    font=dict(color='#1E88E5', size=16),
                    y=0.95
                ),
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.95)',
                margin=dict(l=80, r=80, t=50, b=50)
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a bar chart showing the same data for clarity
            st.subheader("Detailed Metrics Comparison")
            
            # Create bar chart for better visibility
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=radar_data,
                    marker_color='#1E88E5',
                    text=[f'{val:.1f}' for val in radar_data],
                    textposition='auto',
                    textfont=dict(color='#000000', size=12)
                )
            ])
            
            fig_bar.update_layout(
                height=300,
                title=dict(
                    text="Normalized Financial Metrics",
                    font=dict(color='#1E88E5')
                ),
                xaxis=dict(
                    title='Metric Categories',
                    title_font=dict(color='#000000'),
                    tickfont=dict(color='#000000')
                ),
                yaxis=dict(
                    title='Normalized Value',
                    title_font=dict(color='#000000'),
                    tickfont=dict(color='#000000'),
                    range=[0, max(radar_data) * 1.2]
                ),
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.95)',
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.warning(f"⚠️ Visualization error: {e}")
            st.info("Please ensure all input values are valid.")
        
    with col2:
        st.header("🎯 Prediction Result")
        
        # Make prediction button
        if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
            input_df = create_feature_dataframe(input_data)
            prediction, proba = predict_loan(model, scaler, input_df)
            
            # Display prediction
            if prediction == 0:
                st.success(f"✅ APPROVED")
                st.metric("Confidence", f"{proba[0]*100:.1f}%")
                st.info("Based on your financial profile, you qualify for the loan")
            else:
                st.error(f"❌ REJECTED")
                st.metric("Confidence", f"{proba[1]*100:.1f}%")
                st.warning("Please review your financial profile")
            
            # Additional information
            st.markdown("---")
            st.subheader("📌 Decision Factors")
            
            factors = []
            if cibil_score >= 700:
                factors.append("✅ High CIBIL Score (700+)")
            elif cibil_score < 500:
                factors.append("❌ Low CIBIL Score (<500)")
            else:
                factors.append("⚠️ Average CIBIL Score (500-700)")
            
            if loan_to_income_ratio < 200:
                factors.append("✅ Low Loan-to-Income Ratio")
            elif loan_to_income_ratio > 500:
                factors.append("❌ High Loan-to-Income Ratio")
            else:
                factors.append("⚠️ Moderate Loan-to-Income Ratio")
            
            if asset_to_loan_ratio > 150:
                factors.append("✅ Strong Asset Coverage")
            elif asset_to_loan_ratio < 50:
                factors.append("❌ Weak Asset Coverage")
            else:
                factors.append("⚠️ Average Asset Coverage")
            
            for factor in factors:
                st.write(factor)
            
            # Show probability distribution
            st.markdown("---")
            st.subheader("📊 Probability Distribution")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Approved', 'Rejected'],
                    y=[proba[0], proba[1]],
                    marker_color=['#28a745', '#dc3545'],
                    text=[f'{proba[0]:.1%}', f'{proba[1]:.1%}'],
                    textposition='auto',
                    textfont=dict(color='#000000', size=14)
                )
            ])
            fig.update_layout(
                height=300,
                showlegend=False,
                yaxis_title="Probability",
                yaxis=dict(
                    title_font=dict(color='#000000'), 
                    tickfont=dict(color='#000000'),
                    range=[0, 1]
                ),
                xaxis=dict(
                    title_font=dict(color='#000000'), 
                    tickfont=dict(color='#000000')
                ),
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='rgba(255,255,255,0.95)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Click the 'Predict Loan Approval' button to get the prediction result.")
            
            # How to use section
            st.markdown("---")
            st.subheader("ℹ️ How to Use")
            st.markdown("""
                1. Fill in all applicant details in the sidebar
                2. Review the application summary in the main area
                3. Click 'Predict Loan Approval' to get the result
                
                The system analyzes:
                - Credit worthiness (CIBIL Score)
                - Income stability
                - Asset coverage
                - Loan affordability
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
        **🏦 Loan Approval Prediction System**  
        Built with Streamlit & Scikit-Learn  
        ⚠️ Note: This prediction is based on machine learning model and should be used as a reference only.
    """)

if __name__ == "__main__":
    main()