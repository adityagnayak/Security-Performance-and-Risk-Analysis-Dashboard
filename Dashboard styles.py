"""
Custom CSS styles for Security Performance & Risk Analysis Dashboard
Load this file in the main dashboard application
"""

def get_custom_css():
    """Returns the custom CSS as a string"""
    return """
    <style>
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main {
        padding: 0rem 1rem;
    }
    /* Ensure sidebar is visible */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] > div {
        background-color: #1e1e1e;
    }
    /* Sidebar text colors */
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    .stMetric {
        background-color: #2c3e50 !important;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #1f77b4;
    }
    .stMetric label, .stMetric div {
        color: #ffffff !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .metric-card {
        background-color: #2c3e50;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #ffffff;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        font-family: 'Times New Roman', Times, serif !important;
    }
    h2, h3 {
        color: #ffffff;
        font-weight: 500;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stAlert {
        background-color: #2c3e50 !important;
        border-left: 4px solid #1f77b4;
        color: #ffffff !important;
    }
    /* Success messages */
    .stSuccess {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
        border-left: 4px solid #28a745 !important;
    }
    /* Info messages */
    .stInfo {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
        border-left: 4px solid #17a2b8 !important;
    }
    /* Warning messages */
    .stWarning {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
        border-left: 4px solid #ffc107 !important;
    }
    /* Error messages */
    .stError {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
        border-left: 4px solid #dc3545 !important;
    }
    /* Target alert divs */
    [data-testid="stAlert"] {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div {
        color: #ffffff !important;
    }
    p, label, span, div {
        color: #ffffff !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    /* Hide Streamlit branding but keep sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Disclaimer styling */
    .disclaimer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #ff0000;
        color: #ffffff;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 14px;
        z-index: 999;
        border-top: 3px solid #ffffff;
        font-family: 'Times New Roman', Times, serif !important;
    }
    </style>
    """


def get_disclaimer_html():
    """Returns the disclaimer HTML"""
    return """
    <div class="disclaimer">
        ⚠️ DISCLAIMER: This tool has been developed as an educational platform and not for financial advice. 
        Data displayed on this dashboard may be delayed by up to 30 minutes. 
        Please consult a registered financial advisor before making any financial decisions.
    </div>
    """
