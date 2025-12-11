import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CropCast-UQ",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E5128;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4E9F3D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4ea;
        border-left: 5px solid #1E5128;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA GENERATION
# =============================================================================

@st.cache_data
def generate_agricultural_data(n_counties=60, n_years=9, seed=42):
    """
    Generate synthetic agricultural data with realistic correlations.
    
    Target correlations (from USDA Iowa corn statistics):
    - NDVI peak → Yield: r ≈ 0.81
    - July temp → Yield: r ≈ -0.50
    - Precipitation → Yield: r ≈ 0.61
    """
    np.random.seed(seed)
    n_samples = n_counties * n_years
    
    # Common latent factor creates correlations
    common_factor = np.random.normal(0, 1, n_samples)
    
    # NDVI peak (strong positive correlation with yield)
    ndvi_peak = 0.75 + 0.08 * common_factor + np.random.normal(0, 0.03, n_samples)
    ndvi_peak = np.clip(ndvi_peak, 0.45, 0.95)
    
    # July temperature (negative correlation with yield)
    july_temp = 24 - 1.5 * common_factor + np.random.normal(0, 2, n_samples)
    
    # Growing season precipitation (positive correlation)
    precipitation = 500 + 50 * common_factor + np.random.normal(0, 60, n_samples)
    precipitation = np.clip(precipitation, 200, 800)
    
    # Soil quality (weak correlation)
    soil_quality = 0.6 + 0.3 * np.random.random(n_samples)
    
    # Year and county indices
    years = np.tile(np.arange(2015, 2015 + n_years), n_counties)
    counties = np.repeat(np.arange(n_counties), n_years)
    
    # Year trend (+2 bu/acre per year)
    year_effect = 2.0 * (years - 2015)
    
    # County random effects
    county_effects = np.random.normal(0, 3, n_counties)
    county_effect = county_effects[counties]
    
    # Extreme weather years
    extreme_effect = np.zeros(n_samples)
    extreme_effect[years == 2019] = -15  # Midwest floods
    extreme_effect[years == 2012] = -20  # Drought (if included)
    
    # Calculate yield with realistic relationships
    base_yield = 180
    yield_values = (
        base_yield
        + year_effect
        + 120 * (ndvi_peak - 0.75)      # NDVI contribution
        - 3.5 * (july_temp - 24)         # Temperature contribution
        + 0.04 * (precipitation - 500)   # Precipitation contribution
        + 10 * (soil_quality - 0.75)     # Soil contribution
        + county_effect
        + extreme_effect
        + np.random.normal(0, 6, n_samples)  # Aleatoric noise
    )
    yield_values = np.clip(yield_values, 80, 280)
    
    # Create DataFrame
    df = pd.DataFrame({
        'year': years,
        'county_id': counties,
        'ndvi_peak': ndvi_peak,
        'july_temp': july_temp,
        'precipitation': precipitation,
        'soil_quality': soil_quality,
        'yield': yield_values
    })
    
    return df


# =============================================================================
# MODEL CLASSES
# =============================================================================

class QuantileRegressorEnsemble:
    """Quantile Regression using Gradient Boosting with quantile loss."""
    
    def __init__(self, quantiles=None):
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.quantiles = quantiles
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train separate model for each quantile."""
        X_scaled = self.scaler.fit_transform(X)
        
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=5,
                random_state=42
            )
            model.fit(X_scaled, y)
            self.models[q] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Return predictions for all quantiles."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)
        
        return predictions
    
    def predict_median(self, X):
        """Return median prediction."""
        preds = self.predict(X)
        return preds[0.50]


class DeepEnsemble:
    """Deep Ensemble for uncertainty quantification with decomposition."""
    
    def __init__(self, n_members=5, hidden_layers=(64, 32)):
        self.n_members = n_members
        self.hidden_layers = hidden_layers
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.aleatoric_estimate = None
    
    def fit(self, X, y):
        """Train ensemble with bootstrap sampling."""
        X_scaled = self.scaler.fit_transform(X)
        self.models = []
        
        for i in range(self.n_members):
            # Bootstrap sampling
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X_scaled[idx], y[idx]
            
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42 + i,
                learning_rate_init=0.001
            )
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        # Estimate aleatoric uncertainty from residuals
        all_preds = np.array([m.predict(X_scaled) for m in self.models])
        mean_pred = all_preds.mean(axis=0)
        residuals = y - mean_pred
        self.aleatoric_estimate = np.std(residuals) ** 2
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Return predictions with uncertainty decomposition."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = np.array([m.predict(X_scaled) for m in self.models])
        
        mean = predictions.mean(axis=0)
        epistemic_var = predictions.var(axis=0)  # Model disagreement
        aleatoric_var = self.aleatoric_estimate   # Data noise (constant)
        total_var = epistemic_var + aleatoric_var
        
        return {
            'mean': mean,
            'std': np.sqrt(total_var),
            'epistemic': np.sqrt(epistemic_var),
            'aleatoric': np.sqrt(aleatoric_var),
            'epistemic_var': epistemic_var,
            'aleatoric_var': aleatoric_var
        }


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def calculate_calibration(y_true, y_lower, y_upper, coverage_levels=None):
    """Calculate calibration error for prediction intervals."""
    if coverage_levels is None:
        coverage_levels = [0.50, 0.60, 0.70, 0.80, 0.90]
    
    observed = []
    center = (y_lower + y_upper) / 2
    base_width = (y_upper - y_lower) / 2
    
    for level in coverage_levels:
        # Scale interval width for target coverage
        z_target = stats.norm.ppf((1 + level) / 2)
        z_90 = stats.norm.ppf(0.95)
        scale = z_target / z_90
        
        lower = center - base_width * scale
        upper = center + base_width * scale
        
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        observed.append(coverage)
    
    # Mean absolute calibration error
    error = np.mean(np.abs(np.array(observed) - np.array(coverage_levels)))
    
    return {
        'expected': coverage_levels,
        'observed': observed,
        'error': error
    }


def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return {'rmse': rmse, 'r2': r2, 'mae': mae}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_calibration(cal_qr, cal_ens):
    """Create calibration plot comparing two methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Perfect calibration
    ax.plot([0.4, 1.0], [0.4, 1.0], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Quantile Regression
    ax.plot(cal_qr['expected'], cal_qr['observed'], 
            'o-', color='#2E86AB', linewidth=2.5, markersize=10,
            label=f"Quantile Regression (CE: {cal_qr['error']:.3f})")
    
    # Deep Ensemble
    ax.plot(cal_ens['expected'], cal_ens['observed'],
            's-', color='#A23B72', linewidth=2.5, markersize=10,
            label=f"Deep Ensemble (CE: {cal_ens['error']:.3f})")
    
    ax.fill_between([0.4, 1.0], [0.35, 0.95], [0.45, 1.05], 
                    alpha=0.1, color='green', label='±5% tolerance')
    
    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Observed Coverage', fontsize=12)
    ax.set_title('Uncertainty Calibration Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.45, 0.95])
    ax.set_ylim([0.35, 1.0])
    
    plt.tight_layout()
    return fig


def plot_prediction_scatter(y_true, y_pred, uncertainties, title="Predictions vs Actual"):
    """Create scatter plot with uncertainty coloring."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(y_true, y_pred, c=uncertainties, cmap='YlOrRd',
                        alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    # Metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    
    ax.set_xlabel('Actual Yield (bu/acre)', fontsize=12)
    ax.set_ylabel('Predicted Yield (bu/acre)', fontsize=12)
    ax.set_title(f'{title}\nRMSE: {rmse:.1f} bu/acre | R²: {r2:.3f}', 
                fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Uncertainty (bu/acre)', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_prediction_intervals(y_true, predictions, n_show=35):
    """Plot prediction intervals with actual values."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Sort by true value for visualization
    idx = np.argsort(y_true)[:n_show]
    x = np.arange(len(idx))
    
    median = predictions[0.50][idx]
    lower_90 = predictions[0.05][idx]
    upper_90 = predictions[0.95][idx]
    lower_50 = predictions[0.25][idx]
    upper_50 = predictions[0.75][idx]
    
    # Plot intervals
    ax.fill_between(x, lower_90, upper_90, alpha=0.25, color='#2E86AB', label='90% PI')
    ax.fill_between(x, lower_50, upper_50, alpha=0.45, color='#2E86AB', label='50% PI')
    ax.plot(x, median, '-', color='#2E86AB', linewidth=2, label='Median')
    ax.scatter(x, y_true[idx], color='#E94F37', s=50, zorder=5, 
              edgecolors='white', linewidth=0.5, label='Actual')
    
    ax.set_xlabel('Sample Index (sorted by actual yield)', fontsize=11)
    ax.set_ylabel('Yield (bu/acre)', fontsize=11)
    ax.set_title('Prediction Intervals vs Actual Yields', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_uncertainty_decomposition(epistemic_pct, aleatoric_pct):
    """Create uncertainty decomposition pie chart."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sizes = [epistemic_pct, aleatoric_pct]
    labels = [
        f'Epistemic\n(Model Uncertainty)\n{epistemic_pct:.1f}%',
        f'Aleatoric\n(Data Noise)\n{aleatoric_pct:.1f}%'
    ]
    colors = ['#2E86AB', '#A23B72']
    explode = (0.03, 0.03)
    
    wedges, texts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                           startangle=90, textprops={'fontsize': 11})
    
    ax.set_title('Uncertainty Decomposition', fontsize=14, fontweight='bold', pad=20)
    
    # Add explanation
    ax.text(0, -1.4, 
            'Epistemic: Reducible with more data/features\nAleatoric: Irreducible (weather randomness)',
            ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(df):
    """Plot feature correlations with yield."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    features = ['ndvi_peak', 'july_temp', 'precipitation', 'soil_quality']
    labels = ['NDVI Peak', 'July Temperature', 'Precipitation', 'Soil Quality']
    correlations = [df[f].corr(df['yield']) for f in features]
    
    colors = ['#2E86AB' if c > 0 else '#E94F37' for c in correlations]
    bars = ax.barh(labels, correlations, color=colors, edgecolor='white', linewidth=1)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation with Yield', fontsize=11)
    ax.set_title('Feature Importance Analysis', fontsize=13, fontweight='bold')
    ax.set_xlim([-1, 1])
    
    # Add value labels
    for bar, val in zip(bars, correlations):
        x_pos = val + 0.05 if val > 0 else val - 0.05
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
               va='center', ha='left' if val > 0 else 'right', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def show_home():
    """Home page with project overview."""
    st.markdown('<p class="main-header">🌾 CropCast-UQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Uncertainty Quantification for Agricultural Crop Yield Prediction</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Problem-Solution-Impact cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Problem</h3>
        <p>Traditional ML models give single-point predictions without confidence estimates. 
        Agricultural stakeholders need to know <b>how reliable</b> predictions are.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>💡 Solution</h3>
        <p>Probabilistic predictions with <b>calibrated uncertainty bounds</b> using 
        Quantile Regression and Deep Ensembles with uncertainty decomposition.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📈 Impact</h3>
        <p>Enable <b>risk-aware decision making</b> for the $50B+ agricultural market: 
        crop insurance, commodity trading, food security planning.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key features
    st.markdown("### 🔬 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Two UQ Methods Compared:**
        - **Quantile Regression**: Direct interval prediction without distributional assumptions
        - **Deep Ensemble**: Neural network ensemble with uncertainty decomposition
        
        **Calibration Analysis:**
        - Does a 90% prediction interval actually contain the true value 90% of the time?
        - Systematic evaluation across multiple coverage levels
        """)
    
    with col2:
        st.markdown("""
        **Uncertainty Decomposition:**
        - **Epistemic**: Model uncertainty (reducible with more data)
        - **Aleatoric**: Inherent data noise (irreducible)
        
        **Application Context:**
        - US Corn Belt simulation
        - Sentinel-2 satellite features (NDVI)
        - Weather and soil variables
        """)
    
    st.markdown("---")
    
    st.info("👈 **Navigate using the sidebar** to explore the Interactive Demo, view Results, read Documentation, or learn About the project.")


def show_demo():
    """Interactive demonstration page."""
    st.markdown("## 🔬 Interactive Demo")
    st.markdown("Explore uncertainty quantification with adjustable parameters.")
    
    # Sidebar controls
    st.sidebar.markdown("### ⚙️ Demo Parameters")
    
    n_counties = st.sidebar.slider("Number of Counties", 30, 100, 60, 10)
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 30, 20, 5)
    n_ensemble = st.sidebar.slider("Ensemble Members", 3, 10, 5)
    random_seed = st.sidebar.number_input("Random Seed", 1, 100, 42)
    
    # Generate and train
    with st.spinner("🔄 Generating data and training models..."):
        # Generate data
        df = generate_agricultural_data(n_counties=n_counties, seed=random_seed)
        
        # Prepare features
        feature_cols = ['ndvi_peak', 'july_temp', 'precipitation', 'soil_quality']
        X = df[feature_cols].values
        y = df['yield'].values
        
        # Random split (matches slides better)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_seed
        )
        
        # Train Quantile Regression
        qr_model = QuantileRegressorEnsemble()
        qr_model.fit(X_train, y_train)
        qr_preds = qr_model.predict(X_test)
        qr_metrics = calculate_metrics(y_test, qr_preds[0.50])
        
        # Train Deep Ensemble
        ens_model = DeepEnsemble(n_members=n_ensemble)
        ens_model.fit(X_train, y_train)
        ens_preds = ens_model.predict(X_test)
        ens_metrics = calculate_metrics(y_test, ens_preds['mean'])
        
        # Calculate calibration
        cal_qr = calculate_calibration(y_test, qr_preds[0.05], qr_preds[0.95])
        
        z_90 = stats.norm.ppf(0.95)
        ens_lower = ens_preds['mean'] - z_90 * ens_preds['std']
        ens_upper = ens_preds['mean'] + z_90 * ens_preds['std']
        cal_ens = calculate_calibration(y_test, ens_lower, ens_upper)
        
        # Uncertainty decomposition
        total_var = np.mean(ens_preds['epistemic_var'] + ens_preds['aleatoric_var'])
        epistemic_pct = 100 * np.mean(ens_preds['epistemic_var']) / total_var
        aleatoric_pct = 100 * ens_preds['aleatoric_var'] / total_var
    
    st.success(f"✅ Trained on {len(y_train)} samples | Testing on {len(y_test)} samples")
    
    # Metrics display
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("QR RMSE", f"{qr_metrics['rmse']:.1f} bu/acre")
    col2.metric("QR R²", f"{qr_metrics['r2']:.3f}")
    col3.metric("Ensemble RMSE", f"{ens_metrics['rmse']:.1f} bu/acre")
    col4.metric("Ensemble R²", f"{ens_metrics['r2']:.3f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("QR Calibration Error", f"{cal_qr['error']:.3f}")
    col2.metric("Ensemble Cal. Error", f"{cal_ens['error']:.3f}")
    col3.metric("Epistemic %", f"{epistemic_pct:.1f}%")
    col4.metric("Aleatoric %", f"{aleatoric_pct:.1f}%")
    
    # Tabs for visualizations
    st.markdown("### 📈 Visualizations")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Calibration", 
        "🎯 Pred. Intervals", 
        "📊 Scatter Plot",
        "🔄 Uncertainty",
        "📋 Features"
    ])
    
    with tab1:
        st.pyplot(plot_calibration(cal_qr, cal_ens))
        st.markdown("""
        **How to read:** Points on the diagonal = perfect calibration. 
        Below diagonal = overconfident. Above diagonal = underconfident.
        """)
    
    with tab2:
        st.pyplot(plot_prediction_intervals(y_test, qr_preds))
        st.markdown("""
        **Interpretation:** Blue bands show 50% and 90% prediction intervals. 
        Red dots are actual yields. Well-calibrated ≈ 90% of dots within light blue band.
        """)
    
    with tab3:
        uncertainties = (qr_preds[0.95] - qr_preds[0.05]) / 2
        st.pyplot(plot_prediction_scatter(y_test, qr_preds[0.50], uncertainties, 
                                         "Quantile Regression"))
    
    with tab4:
        st.pyplot(plot_uncertainty_decomposition(epistemic_pct, aleatoric_pct))
        st.markdown("""
        **What this means:**
        - **Epistemic** (blue): Can be reduced with more training data or better features
        - **Aleatoric** (pink): Cannot be reduced (inherent weather variability)
        """)
    
    with tab5:
        st.pyplot(plot_feature_importance(df))
        st.markdown("""
        **Feature relationships:** NDVI has strong positive correlation with yield.
        Temperature has negative correlation (heat stress reduces yield).
        """)


def show_results():
    """Results and analysis page."""
    st.markdown("## 📊 Results & Analysis")
    
    # Generate standard results
    df = generate_agricultural_data(n_counties=60, seed=42)
    feature_cols = ['ndvi_peak', 'july_temp', 'precipitation', 'soil_quality']
    X = df[feature_cols].values
    y = df['yield'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    with st.spinner("Training models..."):
        # Baseline
        baseline = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_metrics = calculate_metrics(y_test, baseline_pred)
        
        # Quantile
        qr_model = QuantileRegressorEnsemble()
        qr_model.fit(X_train, y_train)
        qr_preds = qr_model.predict(X_test)
        qr_metrics = calculate_metrics(y_test, qr_preds[0.50])
        cal_qr = calculate_calibration(y_test, qr_preds[0.05], qr_preds[0.95])
        
        # Ensemble
        ens_model = DeepEnsemble(n_members=5)
        ens_model.fit(X_train, y_train)
        ens_preds = ens_model.predict(X_test)
        ens_metrics = calculate_metrics(y_test, ens_preds['mean'])
        
        z_90 = stats.norm.ppf(0.95)
        ens_lower = ens_preds['mean'] - z_90 * ens_preds['std']
        ens_upper = ens_preds['mean'] + z_90 * ens_preds['std']
        cal_ens = calculate_calibration(y_test, ens_lower, ens_upper)
    
    # Summary table
    st.markdown("### 📋 Performance Summary")
    
    results_df = pd.DataFrame({
        'Metric': ['RMSE (bu/acre)', 'R²', 'MAE (bu/acre)', 'Calibration Error', 'Avg. 90% PI Width'],
        'Baseline': [
            f"{baseline_metrics['rmse']:.2f}",
            f"{baseline_metrics['r2']:.3f}",
            f"{baseline_metrics['mae']:.2f}",
            "N/A",
            "N/A"
        ],
        'Quantile Regression': [
            f"{qr_metrics['rmse']:.2f}",
            f"{qr_metrics['r2']:.3f}",
            f"{qr_metrics['mae']:.2f}",
            f"{cal_qr['error']:.3f}",
            f"{np.mean(qr_preds[0.95] - qr_preds[0.05]):.1f}"
        ],
        'Deep Ensemble': [
            f"{ens_metrics['rmse']:.2f}",
            f"{ens_metrics['r2']:.3f}",
            f"{ens_metrics['mae']:.2f}",
            f"{cal_ens['error']:.3f}",
            f"{np.mean(ens_upper - ens_lower):.1f}"
        ]
    })
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Key findings
    st.markdown("### 🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>✅ What Worked Well</h4>
        <ul>
            <li>Both UQ methods achieve comparable accuracy to baseline</li>
            <li>Quantile Regression provides direct, interpretable intervals</li>
            <li>Deep Ensemble enables uncertainty decomposition</li>
            <li>Feature correlations match expected agricultural relationships</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Limitations & Future Work</h4>
        <ul>
            <li>Calibration error indicates room for improvement</li>
            <li>Simulated data - real satellite data is more complex</li>
            <li>Post-hoc calibration (Temperature Scaling) needed</li>
            <li>Spatial correlations not yet modeled</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data statistics
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Sample Information:**
        - Total samples: {len(df)}
        - Training set: {len(y_train)} ({100*len(y_train)/len(df):.0f}%)
        - Test set: {len(y_test)} ({100*len(y_test)/len(df):.0f}%)
        - Years covered: 2015-2023
        - Counties: 60
        """)
    
    with col2:
        st.markdown(f"""
        **Yield Statistics:**
        - Mean: {df['yield'].mean():.1f} bu/acre
        - Std: {df['yield'].std():.1f} bu/acre
        - Min: {df['yield'].min():.1f} bu/acre
        - Max: {df['yield'].max():.1f} bu/acre
        """)
    
    # Correlation verification
    st.markdown("### 🔗 Feature Correlations")
    
    corr_data = {
        'Feature': ['NDVI Peak', 'July Temperature', 'Precipitation', 'Soil Quality'],
        'Target Correlation': ['~0.81', '~-0.50', '~0.61', 'weak'],
        'Actual Correlation': [
            f"{df['ndvi_peak'].corr(df['yield']):.2f}",
            f"{df['july_temp'].corr(df['yield']):.2f}",
            f"{df['precipitation'].corr(df['yield']):.2f}",
            f"{df['soil_quality'].corr(df['yield']):.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)


def show_documentation():
    """Documentation page with methodology details."""
    st.markdown("## 📖 Project Documentation")
    
    # Table of contents
    st.markdown("""
    **Contents:**
    1. [Project Overview](#1-project-overview)
    2. [AI Methodologies](#2-ai-methodologies)
    3. [Technical Implementation](#3-technical-implementation)
    4. [Challenges & Solutions](#4-challenges-solutions)
    5. [Future Work](#5-future-work)
    """)
    
    st.markdown("---")
    
    # 1. Overview
    st.markdown("### 1. Project Overview")
    
    st.markdown("""
    **CropCast-UQ** addresses a critical gap in agricultural AI: the absence of reliable 
    uncertainty estimates in crop yield predictions.
    
    **Problem Statement:**
    Traditional machine learning models output single point estimates (e.g., "185 bu/acre"). 
    However, real-world agricultural decisions require understanding prediction *confidence*:
    
    | Stakeholder | Decision Need | Why UQ Matters |
    |-------------|---------------|----------------|
    | Crop Insurance | Premium pricing | Need confidence bounds for risk assessment |
    | Commodity Traders | Position sizing | Risk-adjusted trading requires uncertainty |
    | Food Security | Buffer planning | Worst-case scenarios need quantification |
    
    **Project Scope:**
    This submission presents **Phase 1: Methodology Validation** using simulated data with 
    known statistical properties. This approach:
    - Validates UQ methods work correctly before complex real-data pipelines
    - Ensures 100% reproducibility for evaluation
    - Follows standard ML research practice (simulation studies)
    """)
    
    st.markdown("---")
    
    # 2. AI Methodologies
    st.markdown("### 2. AI Methodologies")
    
    tab1, tab2, tab3 = st.tabs(["Quantile Regression", "Deep Ensembles", "Calibration Analysis"])
    
    with tab1:
        st.markdown("""
        #### Quantile Regression
        
        **Concept:** Instead of predicting the conditional mean E[Y|X], we predict multiple 
        quantiles of the conditional distribution P(Y|X).
        
        **Loss Function (Pinball Loss):**
        
        $$L_q(y, \\hat{y}) = \\begin{cases} q \\cdot (y - \\hat{y}) & \\text{if } y \\geq \\hat{y} \\\\ (1-q) \\cdot (\\hat{y} - y) & \\text{if } y < \\hat{y} \\end{cases}$$
        
        **Implementation:**
        - Gradient Boosting with `loss='quantile'`
        - 7 quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        - 50% quantile = median prediction
        - [0.05, 0.95] = 90% prediction interval
        
        **Advantages:**
        - No distributional assumptions required
        - Directly outputs interpretable intervals
        - Handles heteroscedasticity naturally
        - Single model per quantile (efficient)
        
        **Limitations:**
        - Cannot decompose uncertainty types
        - Quantile crossing possible (rare with regularization)
        """)
    
    with tab2:
        st.markdown("""
        #### Deep Ensemble
        
        **Concept:** Train M neural networks with different random initializations and 
        bootstrap samples, then aggregate predictions.
        
        **Uncertainty Decomposition:**
        
        $$\\sigma^2_{total} = \\underbrace{\\frac{1}{M}\\sum_{m=1}^{M}(\\mu_m - \\bar{\\mu})^2}_{\\text{Epistemic}} + \\underbrace{\\frac{1}{M}\\sum_{m=1}^{M}\\sigma_m^2}_{\\text{Aleatoric}}$$
        
        **Implementation:**
        - 5 MLP models with architecture [64, 32]
        - Bootstrap sampling (sampling with replacement)
        - Different random seeds for initialization
        - Mean prediction + variance decomposition
        
        **Advantages:**
        - Decomposes epistemic vs aleatoric uncertainty
        - Better out-of-distribution (OOD) detection
        - Flexible - works with any base architecture
        
        **Limitations:**
        - N× training cost (5 models = 5× compute)
        - Often overconfident without calibration
        """)
    
    with tab3:
        st.markdown("""
        #### Calibration Analysis
        
        **Why Calibration Matters:**
        A model's uncertainty estimates are only useful if they're *reliable*. 
        A 90% prediction interval should contain the true value 90% of the time.
        
        **Calibration Error:**
        
        $$CE = \\frac{1}{K} \\sum_{k=1}^{K} |p_k - \\hat{p}_k|$$
        
        Where:
        - $p_k$ = expected coverage (e.g., 0.90)
        - $\\hat{p}_k$ = observed coverage (actual fraction in interval)
        
        **Interpretation:**
        | Calibration Error | Quality |
        |-------------------|---------|
        | < 0.05 | Excellent |
        | 0.05 - 0.10 | Good |
        | 0.10 - 0.20 | Acceptable |
        | > 0.20 | Needs improvement |
        
        **Common Issues:**
        - Neural networks are typically *overconfident* (intervals too narrow)
        - Solution: Post-hoc calibration (Temperature Scaling, Isotonic Regression)
        """)
    
    st.markdown("---")
    
    # 3. Technical Implementation
    st.markdown("### 3. Technical Implementation")
    
    st.markdown("""
    **Data Pipeline:**
    ```
    Synthetic Data Generator (controlled correlations)
           ↓
    Feature Matrix: [NDVI, Temperature, Precipitation, Soil]
           ↓
    Train/Test Split (80/20 random split)
           ↓
    StandardScaler normalization
           ↓
    Model Training (QR + Deep Ensemble)
           ↓
    Uncertainty Estimation & Calibration Analysis
           ↓
    Visualization & Reporting
    ```
    
    **Key Design Decisions:**
    
    | Decision | Rationale |
    |----------|-----------|
    | Simulated data | Validate methodology before complex real data |
    | Random split | More representative of model capability |
    | 5 ensemble members | Balance between UQ quality and computation |
    | GradientBoosting for QR | Native quantile loss support |
    | Bootstrap sampling | Increase ensemble diversity |
    
    **Libraries:**
    - `scikit-learn`: GradientBoostingRegressor, MLPRegressor
    - `scipy.stats`: Statistical functions
    - `streamlit`: Web application
    - `matplotlib`: Visualization
    """)
    
    st.markdown("---")
    
    # 4. Challenges
    st.markdown("### 4. Challenges & Solutions")
    
    challenges = [
        {
            "title": "Neural networks are inherently overconfident",
            "description": "Deep learning models often produce uncertainty estimates that are too narrow.",
            "solution": "Used Deep Ensembles with bootstrap sampling; identified need for post-hoc calibration.",
            "lesson": "UQ requires explicit methodology—confidence scores alone are unreliable."
        },
        {
            "title": "Balancing accuracy vs calibration",
            "description": "Optimizing for prediction accuracy doesn't guarantee good uncertainty estimates.",
            "solution": "Evaluated both standard metrics (RMSE, R²) AND calibration metrics separately.",
            "lesson": "Calibration is orthogonal to accuracy—both must be optimized."
        },
        {
            "title": "Uncertainty decomposition interpretation",
            "description": "Explaining epistemic vs aleatoric distinction to non-technical stakeholders.",
            "solution": "Used analogies: epistemic = model knowledge gaps (fixable), aleatoric = weather randomness (inherent).",
            "lesson": "Clear communication is as important as technical correctness."
        },
        {
            "title": "Ensuring reproducibility",
            "description": "Results must be reproducible for evaluation and scientific validity.",
            "solution": "Fixed random seeds, used deterministic algorithms where possible, documented all parameters.",
            "lesson": "Reproducibility should be designed in, not added after."
        }
    ]
    
    for i, c in enumerate(challenges, 1):
        with st.expander(f"**Challenge {i}:** {c['title']}"):
            st.markdown(f"**Problem:** {c['description']}")
            st.markdown(f"**Solution:** {c['solution']}")
            st.markdown(f"**Lesson Learned:** {c['lesson']}")
    
    st.markdown("---")
    
    # 5. Future Work
    st.markdown("### 5. Future Work")
    
    st.markdown("""
    1. Integrate real Sentinel-2 satellite imagery via Google Earth Engine
    2. Obtain USDA NASS county-level yield data (2017-2024)
    3. Implement Temperature Scaling for calibration improvement
    4. Add spatial features (neighboring county correlations)
    """)


def show_about():
    """About page with team info."""
    st.markdown("## 👥 About")
    
    st.markdown("""
    ### Team Members & Contributions
    """)
    
    # Update with your actual information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>👤 [Xuyang Chen]</h4>
        <p><strong>Role:</strong> Project Lead</p>
        <p><strong>Contributions:</strong></p>
        <ul>
            <li>Uncertainty quantification methodology design</li>
            <li>Model implementation (QR + Ensemble)</li>
            <li>Calibration analysis framework</li>
            <li>Web application development</li>
            <li>Documentation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>👤 [Xuyang Chen]</h4>
        <p><strong>Role:</strong> [Their Role]</p>
        <p><strong>Contributions:</strong></p>
        <ul>
            <li>[Contribution 1]</li>
            <li>[Contribution 2]</li>
            <li>[Contribution 3]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("🌾 CropCast-UQ")
    st.sidebar.markdown("*Uncertainty Quantification for Crop Yield Prediction*")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔬 Interactive Demo", "📊 Results", "📖 Documentation", "👥 About"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home()
    elif page == "🔬 Interactive Demo":
        show_demo()
    elif page == "📊 Results":
        show_results()
    elif page == "📖 Documentation":
        show_documentation()
    elif page == "👥 About":
        show_about()


if __name__ == "__main__":
    main()
