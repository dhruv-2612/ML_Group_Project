import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import plotly.express as px
import plotly.io as pio
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cluster_job_roles(
    df: pd.DataFrame, 
    n_clusters: int = 6, 
    max_features: int = 500
) -> Tuple[pd.DataFrame, Any, List[List[str]]]:
    """
    Performs K-Means clustering on job descriptions.
    
    Args:
        df: DataFrame with 'job_description' column.
        n_clusters: Number of clusters to create.
        max_features: Max features for TF-IDF.
        
    Returns:
        - DataFrame with 'cluster_id' and 'cluster_label' added.
        - KMeans model object.
        - List of top keywords per cluster.
    """
    if df.empty or 'job_description' not in df.columns:
        logger.error("DataFrame is empty or missing 'job_description'.")
        return df, None, []
    
    # Fill NaN descriptions
    docs = df['job_description'].fillna("")
    
    # 1. TF-IDF Vectorization
    logger.info("Vectorizing job descriptions...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(docs)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # 2. K-Means Clustering
    logger.info(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    
    # Assign clusters to DataFrame
    df['cluster_id'] = kmeans.labels_
    
    # 3. Extract Top Keywords per Cluster
    top_keywords = []
    cluster_centers = kmeans.cluster_centers_
    
    # Get top 15 terms for each cluster center
    for i in range(n_clusters):
        # Sort indices of the center vector
        top_indices = cluster_centers[i].argsort()[::-1][:15]
        terms = feature_names[top_indices]
        top_keywords.append(list(terms))
        
    # Create descriptive labels (e.g., "Cluster 0: manager, product")
    cluster_labels = {
        i: f"Cluster {i}: {', '.join(keywords[:2])}" 
        for i, keywords in enumerate(top_keywords)
    }
    df['cluster_label'] = df['cluster_id'].map(cluster_labels)
    
    logger.info("Clustering complete.")
    return df, kmeans, top_keywords

def get_cluster_characteristics(df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
    """
    Analyzes a specific cluster to return its key stats and features.
    """
    cluster_df = df[df['cluster_id'] == cluster_id]
    
    if cluster_df.empty:
        return {}
        
    # 1. Avg Salary (if available)
    avg_salary = cluster_df['salary_mid'].mean() if 'salary_mid' in cluster_df.columns else 0.0
    
    # 2. Top Companies
    top_companies = cluster_df['company'].value_counts().head(5).index.tolist()
    
    # 3. Top Locations
    top_locations = cluster_df['location'].value_counts().head(5).index.tolist()
    
    # 4. Top Skills (aggregating skills_extracted lists)
    all_skills = []
    if 'skills_extracted' in cluster_df.columns:
        # Flatten list of lists, handling potential string representations if read from CSV
        for skills in cluster_df['skills_extracted']:
            if isinstance(skills, list):
                all_skills.extend(skills)
            elif isinstance(skills, str):
                # Simple cleanup if it's a string repr like "['Python', 'SQL']"
                cleaned = skills.strip("[]").replace("'", "").split(", ")
                all_skills.extend([s for s in cleaned if s])
                
    top_skills = pd.Series(all_skills).value_counts().head(10).index.tolist()
    
    return {
        'cluster_id': int(cluster_id),
        'count': len(cluster_df),
        'avg_salary_lpa': round(avg_salary, 2),
        'top_companies': top_companies,
        'top_locations': top_locations,
        'top_skills': top_skills
    }

def analyze_skills_by_role(df: pd.DataFrame, role_filter: str = None) -> pd.DataFrame:
    """
    Analyzes skill frequency, percentage, and average salary for a given role (or all roles).
    """
    if df.empty or 'skills_extracted' not in df.columns:
        logger.warning("DataFrame empty or missing 'skills_extracted'.")
        return pd.DataFrame()
        
    target_df = df.copy()
    if role_filter:
        target_df = target_df[target_df['job_title'].str.contains(role_filter, case=False, na=False)]
        
    if target_df.empty:
        logger.warning(f"No jobs found for role filter: {role_filter}")
        return pd.DataFrame()

    # Expand list of skills into rows
    # Convert string representation to list if needed
    def parse_skills(x):
        if isinstance(x, list): return x
        if isinstance(x, str): return [s.strip(" '\"") for s in x.strip("[]").split(", ") if s]
        return []

    target_df['skills_list'] = target_df['skills_extracted'].apply(parse_skills)
    exploded = target_df.explode('skills_list')
    
    # Group by skill
    skill_stats = exploded.groupby('skills_list').agg(
        count=('job_id', 'count'),
        avg_salary=('salary_mid', 'mean')
    ).reset_index()
    
    skill_stats['percentage'] = (skill_stats['count'] / len(target_df)) * 100
    skill_stats = skill_stats.sort_values('count', ascending=False)
    
    return skill_stats

def identify_emerging_skills(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Identifies emerging skills by comparing current frequencies with a simulated baseline.
    Returns: DataFrame sorted by growth rate.
    """
    current_stats = analyze_skills_by_role(df)
    
    if current_stats.empty:
        return pd.DataFrame()
        
    # Simulate baseline (e.g., last year's data had 50-90% of current volume randomly)
    # In a real app, you'd load actual historical data
    np.random.seed(42)
    current_stats['baseline_count'] = (current_stats['count'] * np.random.uniform(0.5, 0.9, size=len(current_stats))).astype(int)
    
    # Calculate Growth
    current_stats['growth_rate'] = ((current_stats['count'] - current_stats['baseline_count']) / current_stats['baseline_count']) * 100
    
    emerging = current_stats[current_stats['growth_rate'] > 50].sort_values('growth_rate', ascending=False)
    return emerging.head(top_n)

def calculate_skill_salary_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the salary premium associated with each skill.
    """
    if df.empty or 'salary_mid' not in df.columns or 'skills_extracted' not in df.columns:
        return pd.DataFrame()
        
    # Get list of all unique skills
    def parse_skills(x):
        if isinstance(x, list): return x
        if isinstance(x, str): return [s.strip(" '\"") for s in x.strip("[]").split(", ") if s]
        return []
        
    all_skills = set([s for sublist in df['skills_extracted'].apply(parse_skills) for s in sublist])
    
    impact_data = []
    
    for skill in all_skills:
        # Split population
        with_skill = df[df['skills_extracted'].apply(lambda x: skill in str(x))]['salary_mid'].dropna()
        without_skill = df[~df['skills_extracted'].apply(lambda x: skill in str(x))]['salary_mid'].dropna()
        
        if len(with_skill) < 5 or len(without_skill) < 5:
            continue
            
        avg_with = with_skill.mean()
        avg_without = without_skill.mean()
        premium = avg_with - avg_without
        
        # T-test
        t_stat, p_val = stats.ttest_ind(with_skill, without_skill, equal_var=False)
        
        impact_data.append({
            'skill': skill,
            'avg_salary_with': round(avg_with, 2),
            'avg_salary_without': round(avg_without, 2),
            'premium': round(premium, 2),
            'p_value': round(p_val, 4)
        })
        
    if not impact_data:
        return pd.DataFrame(columns=['skill', 'avg_salary_with', 'avg_salary_without', 'premium', 'p_value'])
        
    return pd.DataFrame(impact_data).sort_values('premium', ascending=False)

def generate_skills_report(df: pd.DataFrame, role: str = "Product Manager") -> Dict[str, Any]:
    """
    Generates a comprehensive skills report for a specific role.
    """
    report = {}
    
    # 1. Skill Analysis
    skill_stats = analyze_skills_by_role(df, role_filter=role)
    report['top_skills'] = skill_stats.head(10).to_dict('records')
    
    # 2. Salary Impact (global for now, to ensure sample size)
    impact_df = calculate_skill_salary_impact(df)
    report['salary_premiums'] = impact_df.head(5).to_dict('records')
    
    # 3. Visualization
    if not skill_stats.empty:
        fig = px.bar(
            skill_stats.head(10),
            x='count',
            y='skills_list',
            orientation='h',
            title=f"Top Skills for {role}",
            labels={'skills_list': 'Skill', 'count': 'Job Count'},
            template='plotly_white'
        )
        output_path = f"data/processed/skills_viz_{role.lower().replace(' ', '_')}.html"
        fig.write_html(output_path)
        report['viz_path'] = output_path
        
    return report

def train_salary_predictor(df: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame, Dict, List[str]]:
    """
    Trains a Random Forest model with advanced feature engineering and comprehensive evaluation.
    """
    if df.empty or 'salary_mid' not in df.columns:
        logger.error("DataFrame empty or missing target 'salary_mid'.")
        return None, pd.DataFrame(), {}, []
        
    # Drop rows with missing salary
    train_df = df.dropna(subset=['salary_mid']).copy()
    
    # --- Feature Engineering ---
    
    # 1. Experience & Skills Count
    X = train_df[['experience_min', 'num_skills']].copy()
    
    # 2. Tech Skills & MBA
    # Define tech keywords (simplified list for feature extraction)
    tech_keywords = ['python', 'sql', 'tableau', 'power bi', 'machine learning', 'aws', 'java', 'react']
    
    def check_tech(text):
        if not isinstance(text, str): return 0
        text = text.lower()
        return 1 if any(k in text for k in tech_keywords) else 0
        
    def check_mba(text):
        if not isinstance(text, str): return 0
        return 1 if 'mba' in text.lower() or 'master' in text.lower() else 0

    # Combine title and description for check
    train_df['full_text'] = train_df['job_title'] + " " + train_df['job_description'].fillna("")
    X['has_technical_skills'] = train_df['full_text'].apply(check_tech)
    X['has_mba'] = train_df['full_text'].apply(check_mba)
    
    # 3. Experience Level (Ordinal Encoding)
    exp_map = {'Entry Level': 0, 'Mid Level': 1, 'Senior Level': 2, 'Lead/Principal': 3}
    X['experience_level_enc'] = train_df['experience_level'].map(exp_map).fillna(0)
    
    # 4. Location Encoding (One-Hot)
    top_locs = train_df['location'].value_counts().head(5).index
    train_df['loc_encoded'] = train_df['location'].apply(lambda x: x if x in top_locs else 'Other')
    loc_dummies = pd.get_dummies(train_df['loc_encoded'], prefix='loc')
    X = pd.concat([X, loc_dummies], axis=1)
    
    # 5. Cluster ID
    if 'cluster_id' in train_df.columns:
        X['cluster'] = train_df['cluster_id']
    else:
        X['cluster'] = 0
        
    y = train_df['salary_mid']
    feature_cols = X.columns.tolist()
    
    # --- Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # --- Evaluation ---
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross Validation (5-fold)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    logger.info(f"Model Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, CV R2: {cv_scores.mean():.2f}")
    
    metrics = {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2_score': round(r2, 2),
        'cv_r2_mean': round(cv_scores.mean(), 2),
        'n_samples': len(train_df)
    }
    
    # --- Feature Importance ---
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # --- Artifacts ---
    # 1. Save Model
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(model, "data/processed/salary_model.pkl")
    joblib.dump(feature_cols, "data/processed/salary_features.pkl") # Save column order
    logger.info("Saved model to data/processed/salary_model.pkl")
    
    # 2. Actual vs Predicted Plot
    fig = px.scatter(
        x=y_test, y=y_pred, 
        labels={'x': 'Actual Salary (LPA)', 'y': 'Predicted Salary (LPA)'},
        title='Actual vs Predicted Salary',
        template='plotly_white'
    )
    fig.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
    fig.write_html("data/processed/salary_prediction_plot.html")
    
    return model, feat_imp, metrics, feature_cols

def predict_salary(model: RandomForestRegressor, profile: Dict[str, Any], feature_columns: List[str]) -> Tuple[float, Tuple[float, float]]:
    """
    Predicts salary range for a user profile.
    Returns: (predicted_salary, (lower_bound, upper_bound))
    """
    # Initialize all features to 0
    input_data = {col: 0 for col in feature_columns}
    
    # Set numeric values
    input_data['experience_min'] = profile.get('years_experience', 0)
    input_data['num_skills'] = len(profile.get('skills', []))
    input_data['cluster'] = profile.get('cluster_id', 0)
    
    # Derived features
    # Tech skills check (simple heuristic for prediction time)
    tech_keywords = ['python', 'sql', 'tableau', 'power bi', 'machine learning', 'aws', 'java', 'react']
    skills_text = " ".join(profile.get('skills', [])).lower()
    input_data['has_technical_skills'] = 1 if any(k in skills_text for k in tech_keywords) else 0
    
    input_data['has_mba'] = 1 if 'mba' in str(profile.get('education', '')).lower() else 0
    
    # Exp level encoding
    exp_years = profile.get('years_experience', 0)
    if exp_years < 2: level = 0
    elif exp_years < 5: level = 1
    elif exp_years < 10: level = 2
    else: level = 3
    input_data['experience_level_enc'] = level
    
    # Set location one-hot
    loc = profile.get('location', 'Other')
    loc_col = f"loc_{loc}"
    
    # Only set 1 if the location column exists in training features
    if loc_col in input_data:
        input_data[loc_col] = 1
    elif 'loc_Other' in input_data:
        input_data['loc_Other'] = 1
        
    # Convert to DataFrame with strict column order
    df_in = pd.DataFrame([input_data])[feature_columns]
    
    # Predict
    pred = model.predict(df_in)[0]
    
    # Heuristic confidence interval (e.g., +/- 15%)
    lower = pred * 0.85
    upper = pred * 1.15
    
    return round(pred, 2), (round(lower, 2), round(upper, 2))

def plot_clusters_visualization(df: pd.DataFrame, title: str = "Job Role Clusters"):
    """
    Visualizes clusters using PCA (2D) and saves to HTML.
    """
    if 'cluster_id' not in df.columns:
        logger.error("Clusters not found. Run cluster_job_roles first.")
        return

    logger.info("Generating PCA visualization...")
    
    # Re-vectorize for PCA (or pass the matrix if we refactored, but this is cleaner for standalone)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    matrix = tfidf.fit_transform(df['job_description'].fillna(""))
    
    # PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(matrix.toarray())
    
    df['pca_x'] = components[:, 0]
    df['pca_y'] = components[:, 1]
    
    # Plotly Scatter
    fig = px.scatter(
        df,
        x='pca_x',
        y='pca_y',
        color='cluster_label',
        hover_data=['job_title', 'company', 'location'],
        title=title,
        template='plotly_white',
        width=1000,
        height=700
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    
    output_path = "data/processed/clusters_viz.html"
    fig.write_html(output_path)
    logger.info(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Test run with synthetic cleaned data
    try:
        df = pd.read_csv("data/processed/cleaned_jobs.csv")
        logger.info(f"Loaded {len(df)} records.")
        
        # 1. Clustering
        clustered_df, kmeans_model, keywords = cluster_job_roles(df, n_clusters=5)
        
        # 2. Skills Analysis for Product Managers
        pm_report = generate_skills_report(clustered_df, role="Product Manager")
        print("\nPM Skills Report:", pm_report)
        
        # 3. Emerging Skills
        emerging = identify_emerging_skills(clustered_df)
        print("\nEmerging Skills (Simulated):")
        print(emerging[['skills_list', 'growth_rate']].head())
        
        # 4. Salary Impact
        impact = calculate_skill_salary_impact(clustered_df)
        print("\nTop Salary Premium Skills:")
        print(impact[['skill', 'premium', 'p_value']].head())
        
        # 5. Salary Model Training
        sal_model, feat_imp, metrics, feature_cols = train_salary_predictor(clustered_df)
        print("\nSalary Model Metrics:", metrics)
        print("Feature Importance:\n", feat_imp.head())
        
        # 6. Prediction Test
        test_profile = {
            'years_experience': 4.0,
            'skills': ['Python', 'SQL', 'Product Management'],
            'education': 'MBA',
            'location': 'Bangalore',
            'cluster_id': 0 
        }
        
        pred, interval = predict_salary(sal_model, test_profile, feature_cols)
        print(f"\nPredicted Salary for {test_profile['years_experience']}y exp in {test_profile['location']}: {pred} LPA (Range: {interval})")
        
        # Visualize
        plot_clusters_visualization(clustered_df)
        clustered_df.to_csv("data/processed/job_clusters.csv", index=False)
        print("Saved clustered data to data/processed/job_clusters.csv")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
