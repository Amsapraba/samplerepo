import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="HDFS Unsupervised ML Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess HDFS log data"""
    # Create synthetic HDFS log data since we don't have the actual file
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic HDFS log features
    block_ids = [f"blk_{np.random.randint(1000000, 9999999)}" for _ in range(n_samples)]
    node_ids = [f"node_{np.random.randint(1, 20)}" for _ in range(n_samples)]
    
    # Log message types
    message_types = ['INFO', 'ERROR', 'WARN', 'DEBUG', 'FATAL']
    log_messages = np.random.choice(message_types, n_samples, p=[0.6, 0.15, 0.15, 0.08, 0.02])
    
    # Numerical features
    data = {
        'block_id': block_ids,
        'node_id': node_ids,
        'message_type': log_messages,
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'file_size_mb': np.random.exponential(100, n_samples),
        'replication_factor': np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.7, 0.2]),
        'response_time_ms': np.random.gamma(2, 50, n_samples),
        'cpu_usage': np.random.beta(2, 5, n_samples) * 100,
        'memory_usage': np.random.beta(3, 2, n_samples) * 100,
        'disk_io': np.random.exponential(200, n_samples),
        'network_io': np.random.exponential(150, n_samples),
        'error_count': np.random.poisson(2, n_samples),
        'session_duration': np.random.exponential(300, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[anomaly_indices, 'response_time_ms'] *= 10
    df.loc[anomaly_indices, 'cpu_usage'] = np.random.uniform(90, 100, len(anomaly_indices))
    
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess data for ML models"""
    # Encode categorical variables
    le_node = LabelEncoder()
    le_message = LabelEncoder()
    
    df_processed = df.copy()
    df_processed['node_id_encoded'] = le_node.fit_transform(df['node_id'])
    df_processed['message_type_encoded'] = le_message.fit_transform(df['message_type'])
    df_processed['hour'] = df['timestamp'].dt.hour
    df_processed['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Select numerical features for clustering
    numerical_features = [
        'file_size_mb', 'replication_factor', 'response_time_ms',
        'cpu_usage', 'memory_usage', 'disk_io', 'network_io',
        'error_count', 'session_duration', 'node_id_encoded',
        'message_type_encoded', 'hour', 'day_of_week'
    ]
    
    X = df_processed[numerical_features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, numerical_features, le_node, le_message

def perform_clustering(X_scaled, n_clusters=5):
    """Perform different clustering algorithms"""
    results = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results['kmeans'] = kmeans.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    results['dbscan'] = dbscan.fit_predict(X_scaled)
    
    # Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    results['agglomerative'] = agg_clustering.fit_predict(X_scaled)
    
    return results

def perform_dimensionality_reduction(X_scaled):
    """Perform dimensionality reduction"""
    results = {}
    
    # PCA
    pca = PCA(n_components=2)
    results['pca'] = pca.fit_transform(X_scaled)
    results['pca_explained_variance'] = pca.explained_variance_ratio_
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    results['tsne'] = tsne.fit_transform(X_scaled[:1000])  # Limit for performance
    
    # SVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    results['svd'] = svd.fit_transform(X_scaled)
    
    return results

def detect_anomalies(X_scaled):
    """Detect anomalies using different methods"""
    results = {}
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    results['isolation_forest'] = iso_forest.fit_predict(X_scaled)
    
    # Elliptic Envelope
    elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
    results['elliptic_envelope'] = elliptic.fit_predict(X_scaled)
    
    return results

def main():
    st.markdown('<h1 class="main-header">🚀 HDFS Unsupervised ML Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Machine Learning for Hadoop Distributed File System Logs**")
    
    # Load data
    with st.spinner("Loading HDFS data..."):
        df = load_data()
        X_scaled, X, feature_names, le_node, le_message = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Model Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🎯 Clustering Analysis", 
        "📉 Dimensionality Reduction",
        "⚠️ Anomaly Detection",
        "📈 Performance Metrics"
    ])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-container"><h3>{len(df):,}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><h3>{len(df.columns)}</h3><p>Features</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-container"><h3>{df["node_id"].nunique()}</h3><p>Unique Nodes</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-container"><h3>{df["message_type"].nunique()}</h3><p>Message Types</p></div>', unsafe_allow_html=True)
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Data Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='message_type', title='Message Type Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y='response_time_ms', title='Response Time Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🎯 Clustering Analysis")
        
        with st.spinner("Performing clustering analysis..."):
            clustering_results = perform_clustering(X_scaled, n_clusters)
        
        col1, col2 = st.columns(2)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        with col1:
            st.subheader("K-Means Clustering")
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], 
                color=clustering_results['kmeans'].astype(str),
                title=f'K-Means Clustering (k={n_clusters})',
                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            cluster_stats = pd.DataFrame({
                'Cluster': range(n_clusters),
                'Count': [np.sum(clustering_results['kmeans'] == i) for i in range(n_clusters)]
            })
            st.dataframe(cluster_stats, use_container_width=True)
        
        with col2:
            st.subheader("DBSCAN Clustering")
            dbscan_labels = clustering_results['dbscan']
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], 
                color=dbscan_labels.astype(str),
                title=f'DBSCAN Clustering ({n_clusters_dbscan} clusters)',
                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # DBSCAN statistics
            unique_labels = np.unique(dbscan_labels)
            dbscan_stats = pd.DataFrame({
                'Cluster': unique_labels,
                'Count': [np.sum(dbscan_labels == label) for label in unique_labels]
            })
            st.dataframe(dbscan_stats, use_container_width=True)
        
        st.subheader("Cluster Characteristics")
        
        # Add cluster labels to original dataframe
        df_with_clusters = df.copy()
        df_with_clusters['kmeans_cluster'] = clustering_results['kmeans']
        
        # Show cluster characteristics
        cluster_chars = df_with_clusters.groupby('kmeans_cluster').agg({
            'response_time_ms': ['mean', 'std'],
            'cpu_usage': ['mean', 'std'],
            'memory_usage': ['mean', 'std'],
            'error_count': ['mean', 'std']
        }).round(2)
        
        st.dataframe(cluster_chars, use_container_width=True)
    
    with tab3:
        st.header("📉 Dimensionality Reduction")
        
        with st.spinner("Performing dimensionality reduction..."):
            dim_red_results = perform_dimensionality_reduction(X_scaled)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PCA Analysis")
            X_pca = dim_red_results['pca']
            explained_var = dim_red_results['pca_explained_variance']
            
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=df['message_type'],
                title=f'PCA (Explained Variance: {explained_var.sum():.2%})',
                labels={'x': f'PC1 ({explained_var[0]:.2%})', 'y': f'PC2 ({explained_var[1]:.2%})'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("t-SNE Analysis")
            X_tsne = dim_red_results['tsne']
            
            fig = px.scatter(
                x=X_tsne[:, 0], y=X_tsne[:, 1],
                color=df['message_type'][:1000],  # Match the subset used in t-SNE
                title='t-SNE Visualization',
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Importance Analysis")
        
        # PCA component analysis
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Plot cumulative explained variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumsum_var) + 1)),
            y=cumsum_var,
            mode='lines+markers',
            name='Cumulative Explained Variance'
        ))
        fig.update_layout(
            title='PCA Cumulative Explained Variance',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance Ratio',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("⚠️ Anomaly Detection")
        
        with st.spinner("Detecting anomalies..."):
            anomaly_results = detect_anomalies(X_scaled)
        
        col1, col2 = st.columns(2)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        with col1:
            st.subheader("Isolation Forest")
            iso_anomalies = anomaly_results['isolation_forest']
            anomaly_labels = ['Normal' if x == 1 else 'Anomaly' for x in iso_anomalies]
            
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=anomaly_labels,
                title='Isolation Forest Anomaly Detection',
                labels={'x': 'PC1', 'y': 'PC2'},
                color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            n_anomalies = np.sum(iso_anomalies == -1)
            st.metric("Anomalies Detected", f"{n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")
        
        with col2:
            st.subheader("Elliptic Envelope")
            elliptic_anomalies = anomaly_results['elliptic_envelope']
            anomaly_labels_elliptic = ['Normal' if x == 1 else 'Anomaly' for x in elliptic_anomalies]
            
            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=anomaly_labels_elliptic,
                title='Elliptic Envelope Anomaly Detection',
                labels={'x': 'PC1', 'y': 'PC2'},
                color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            n_anomalies_elliptic = np.sum(elliptic_anomalies == -1)
            st.metric("Anomalies Detected", f"{n_anomalies_elliptic} ({n_anomalies_elliptic/len(df)*100:.1f}%)")
        
        st.subheader("Anomaly Analysis")
        
        # Analyze anomalies
        df_anomalies = df.copy()
        df_anomalies['is_anomaly_iso'] = iso_anomalies == -1
        df_anomalies['is_anomaly_elliptic'] = elliptic_anomalies == -1
        
        # Show anomalous records
        anomalous_records = df_anomalies[df_anomalies['is_anomaly_iso']]
        if len(anomalous_records) > 0:
            st.write("Sample Anomalous Records (Isolation Forest):")
            st.dataframe(anomalous_records.head(10), use_container_width=True)
        
        # Anomaly characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Time Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[iso_anomalies == 1]['response_time_ms'],
                name='Normal',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df[iso_anomalies == -1]['response_time_ms'],
                name='Anomaly',
                opacity=0.7
            ))
            fig.update_layout(
                title='Response Time: Normal vs Anomalous',
                xaxis_title='Response Time (ms)',
                yaxis_title='Count',
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("CPU Usage Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[iso_anomalies == 1]['cpu_usage'],
                name='Normal',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df[iso_anomalies == -1]['cpu_usage'],
                name='Anomaly',
                opacity=0.7
            ))
            fig.update_layout(
                title='CPU Usage: Normal vs Anomalous',
                xaxis_title='CPU Usage (%)',
                yaxis_title='Count',
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📈 Performance Metrics")
        
        # Calculate silhouette scores for clustering
        from sklearn.metrics import silhouette_score
        
        clustering_metrics = {}
        for name, labels in clustering_results.items():
            if name == 'dbscan' and len(set(labels)) <= 1:
                clustering_metrics[name] = "N/A (insufficient clusters)"
            else:
                try:
                    score = silhouette_score(X_scaled, labels)
                    clustering_metrics[name] = f"{score:.3f}"
                except:
                    clustering_metrics[name] = "N/A"
        
        st.subheader("Clustering Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("K-Means Silhouette Score", clustering_metrics['kmeans'])
        with col2:  
            st.metric("DBSCAN Silhouette Score", clustering_metrics['dbscan'])
        with col3:
            st.metric("Agglomerative Silhouette Score", clustering_metrics['agglomerative'])
        
        st.subheader("Model Comparison")
        
        # Create comparison chart
        comparison_data = {
            'Algorithm': ['K-Means', 'DBSCAN', 'Agglomerative'],
            'Silhouette Score': [
                float(clustering_metrics['kmeans']) if clustering_metrics['kmeans'] != "N/A" else 0,
                0 if clustering_metrics['dbscan'] == "N/A (insufficient clusters)" else float(clustering_metrics['dbscan']) if clustering_metrics['dbscan'] != "N/A" else 0,
                float(clustering_metrics['agglomerative']) if clustering_metrics['agglomerative'] != "N/A" else 0
            ],
            'Clusters Found': [
                n_clusters,
                len(set(clustering_results['dbscan'])) - (1 if -1 in clustering_results['dbscan'] else 0),
                n_clusters
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df, 
            x='Algorithm', 
            y='Silhouette Score',
            title='Clustering Algorithm Performance Comparison',
            color='Silhouette Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Data Quality Report")
        
        quality_metrics = {
            'Total Records': len(df),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Records': df.duplicated().sum(),
            'Data Completeness': f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%",
            'Anomaly Rate (Isolation Forest)': f"{np.sum(anomaly_results['isolation_forest'] == -1) / len(df) * 100:.1f}%"
        }
        
        quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(quality_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 HDFS Unsupervised ML Analysis** - Built with Streamlit & Scikit-learn")

if __name__ == "__main__":
    main()
