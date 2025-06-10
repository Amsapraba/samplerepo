import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Set a title for the Streamlit app
st.title("Simple Unsupervised ML Model with KMeans")
st.write("This app performs KMeans clustering on the `HDFS_2k.log_structured.csv` dataset.")

# Load the dataset
# Assuming the file is in the same directory as the script or accessible via path
file_path = "HDFS_2k.log_structured.csv"

# Check if the file exists
if not os.path.exists(file_path):
    st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as your Streamlit app.")
else:
    try:
        df = pd.read_csv(file_path)
        st.success(f"Successfully loaded '{file_path}'")
        st.write("Original Data (first 5 rows):")
        st.dataframe(df.head())

        # Identify numerical columns for clustering
        # For simplicity, let's try to pick a few numerical columns.
        # In a real scenario, you'd inspect the data more carefully.
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numerical_cols:
            st.warning("No numerical columns found for clustering. Please ensure your CSV contains numerical data or adjust column selection.")
        else:
            st.write(f"Numerical columns selected for clustering: {numerical_cols}")

            # Select features for clustering
            X = df[numerical_cols]

            # Handle missing values by filling with the mean (a simple approach)
            X = X.fillna(X.mean())

            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- Unsupervised Learning: KMeans Clustering ---
            st.header("KMeans Clustering")

            # Allow user to choose number of clusters
            n_clusters = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

            if st.button("Run KMeans Clustering"):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robust centroid initialization
                    clusters = kmeans.fit_predict(X_scaled)
                    df['Cluster'] = clusters

                    st.subheader("Clustering Results")
                    st.write("Data with assigned clusters (first 10 rows):")
                    st.dataframe(df.head(10))

                    st.write("Cluster Distribution:")
                    st.write(df['Cluster'].value_counts().sort_index())

                    st.subheader("Cluster Centroids (Scaled Features)")
                    centroids_scaled_df = pd.DataFrame(kmeans.cluster_centers_, columns=numerical_cols)
                    st.dataframe(centroids_scaled_df)

                    # Optional: Inverse transform centroids to original scale (if meaningful)
                    # centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
                    # st.subheader("Cluster Centroids (Original Feature Scale - Approximation)")
                    # st.dataframe(pd.DataFrame(centroids_original_scale, columns=numerical_cols))

                except Exception as e:
                    st.error(f"An error occurred during KMeans clustering: {e}")
                    st.info("Ensure the selected numerical columns have sufficient variation for clustering.")

    except Exception as e:
        st.error(f"An error occurred while loading or processing the CSV file: {e}")
        st.info("Please check the CSV file format and content.")
