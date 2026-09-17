import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('trainer')

RECOMMENDATION_CATALOG = [
    {'id': 'MOV-01', 'title': 'Cyberstrike: Protocol Zero', 'genres': ['Action', 'Sci-Fi'], 'type': 'Movie', 'duration_mins': 118, 'popularity': 95},
    {'id': 'MOV-02', 'title': 'Velocity: Tokyo Driftline', 'genres': ['Action', 'Thriller'], 'type': 'Movie', 'duration_mins': 112, 'popularity': 92},
    {'id': 'MOV-03', 'title': 'Shadow Operative', 'genres': ['Action', 'Thriller'], 'type': 'Movie', 'duration_mins': 125, 'popularity': 90},
    {'id': 'MOV-04', 'title': 'The Quantum Paradox', 'genres': ['Sci-Fi', 'Thriller'], 'type': 'Movie', 'duration_mins': 135, 'popularity': 88},
    {'id': 'SER-01', 'title': 'Silicon Alchemists', 'genres': ['Drama', 'Sci-Fi'], 'type': 'Series', 'duration_mins': 55, 'popularity': 96},
    {'id': 'SER-02', 'title': 'Crown & Treason', 'genres': ['Drama', 'Thriller'], 'type': 'Series', 'duration_mins': 50, 'popularity': 94},
    {'id': 'SER-03', 'title': 'Midnight In Amsterdam', 'genres': ['Drama', 'Romance'], 'type': 'Series', 'duration_mins': 48, 'popularity': 86},
    {'id': 'SHT-01', 'title': 'Coffee Break Chronicles', 'genres': ['Comedy'], 'type': 'Short', 'duration_mins': 15, 'popularity': 89},
    {'id': 'SHT-02', 'title': 'Office Pet Panic', 'genres': ['Comedy', 'Animation'], 'type': 'Short', 'duration_mins': 18, 'popularity': 87},
    {'id': 'MOV-05', 'title': 'Love in Kyoto', 'genres': ['Romance', 'Comedy'], 'type': 'Movie', 'duration_mins': 98, 'popularity': 85},
    {'id': 'DOC-01', 'title': 'Ocean Depths: Untold Mysteries', 'genres': ['Documentary'], 'type': 'Movie', 'duration_mins': 88, 'popularity': 91},
    {'id': 'DOC-02', 'title': 'AI: The Next Frontier', 'genres': ['Documentary', 'Sci-Fi'], 'type': 'Movie', 'duration_mins': 76, 'popularity': 93},
    {'id': 'MOV-06', 'title': 'Haunted Whisper', 'genres': ['Horror', 'Thriller'], 'type': 'Movie', 'duration_mins': 104, 'popularity': 84},
    {'id': 'SHT-03', 'title': 'Pixel Pals', 'genres': ['Animation', 'Comedy'], 'type': 'Short', 'duration_mins': 12, 'popularity': 88},
    {'id': 'MOV-07', 'title': 'The Grand Heist', 'genres': ['Action', 'Comedy'], 'type': 'Movie', 'duration_mins': 110, 'popularity': 97},
]

ALL_GENRES = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller', 'Romance', 'Documentary', 'Animation', 'Horror']
NUM_COLS = ['watch_time_hours', 'avg_session_mins', 'num_sessions', 'genre_diversity', 'weekend_watch_ratio', 'days_since_last_active']
GENRE_COLS = [f'genre_{g}' for g in ALL_GENRES]
ALL_FEATURE_COLS = NUM_COLS + GENRE_COLS

def find_dataset_path():
    env_path = os.environ.get('DATASET_PATH')
    candidate_paths = [
        env_path,
        'data/user_activity.csv',
        '/app/data/user_activity.csv',
        '../data/user_activity.csv',
    ]
    for p in candidate_paths:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError('Could not find user_activity.csv in any expected location.')

def clean_and_prepare_data(raw_df: pd.DataFrame):
    initial_len = len(raw_df)
    logger.info(f'Loaded raw dataset with {initial_len} records.')

    # 1. Deduplicate by user_id
    df = raw_df.drop_duplicates(subset=['user_id'], keep='first').copy()
    dupes_dropped = initial_len - len(df)
    if dupes_dropped > 0:
        logger.info(f'Dropped {dupes_dropped} duplicate user records.')

    # 2. Filter corrupt sensor / negative values
    invalid_mask = (df['watch_time_hours'] < 0) | (df['avg_session_mins'] < 0)
    neg_dropped = invalid_mask.sum()
    if neg_dropped > 0:
        logger.info(f'Dropped {neg_dropped} records with negative numeric values.')
        df = df[~invalid_mask].copy()

    # 3. Filter extreme impossible outliers
    outlier_mask = (df['watch_time_hours'] > 1000) | (df['avg_session_mins'] > 500)
    outliers_dropped = outlier_mask.sum()
    if outliers_dropped > 0:
        logger.info(f'Dropped {outliers_dropped} extreme outlier records.')
        df = df[~outlier_mask].copy()

    # 4. Impute missing values with contextually appropriate statistics
    watch_median = float(df['watch_time_hours'].median())
    session_median = float(df['avg_session_mins'].median())
    num_sessions_median = int(df['num_sessions'].median())
    weekend_median = float(df['weekend_watch_ratio'].median())
    recency_median = int(df['days_since_last_active'].median())

    df['watch_time_hours'] = df['watch_time_hours'].fillna(watch_median)
    df['avg_session_mins'] = df['avg_session_mins'].fillna(session_median)
    df['num_sessions'] = df['num_sessions'].fillna(num_sessions_median)
    df['weekend_watch_ratio'] = df['weekend_watch_ratio'].fillna(weekend_median)
    df['days_since_last_active'] = df['days_since_last_active'].fillna(recency_median)
    df['genre_diversity'] = df['genre_diversity'].fillna(1).astype(int)

    df['top_genres'] = df['top_genres'].fillna('General').astype(str)
    df['top_genres'] = df['top_genres'].apply(lambda x: 'General' if not x.strip() else x.strip())

    for g in ALL_GENRES:
        df[f'genre_{g}'] = df['top_genres'].apply(lambda x: 1.0 if g.lower() in x.lower() else 0.0)

    clean_len = len(df)
    logger.info(f'Data cleaning complete. Retained {clean_len}/{initial_len} valid records ({clean_len/initial_len:.1%}).')

    defaults = {
        'watch_time_hours': watch_median,
        'avg_session_mins': session_median,
        'num_sessions': num_sessions_median,
        'genre_diversity': 1,
        'weekend_watch_ratio': weekend_median,
        'days_since_last_active': recency_median,
    }
    return df, defaults

def evaluate_cluster_counts(X_full: np.ndarray, min_k: int = 2, max_k: int = 6):
    logger.info(f'Comparing cluster counts from K={min_k} to K={max_k}...')
    results = []
    best_k = min_k
    best_sil = -1.0

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_full)
        sil = float(silhouette_score(X_full, labels))
        db = float(davies_bouldin_score(X_full, labels))
        ch = float(calinski_harabasz_score(X_full, labels))
        inertia = float(km.inertia_)
        counts = {int(cluster_id): int(count) for cluster_id, count in pd.Series(labels).value_counts().items()}

        logger.info(f'K={k} | Silhouette: {sil:.4f} | Inertia: {inertia:.1f} | Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:.1f}')
        results.append({
            'k': k,
            'silhouette_score': round(sil, 4),
            'inertia': round(inertia, 2),
            'davies_bouldin': round(db, 4),
            'calinski_harabasz': round(ch, 2),
            'cluster_sizes': counts
        })

        if sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info(f'Optimal cluster count determined by evidence: K={best_k} (Silhouette = {best_sil:.4f})')
    return best_k, results

def generate_segment_names(df_clean: pd.DataFrame, cluster_labels: np.ndarray, k: int):
    df_eval = df_clean.copy()
    df_eval['cluster'] = cluster_labels
    profiles = {}
    segment_names = {}

    for c in range(k):
        sub = df_eval[df_eval['cluster'] == c]
        count = len(sub)
        pct = count / len(df_eval)

        mean_watch = float(sub['watch_time_hours'].mean())
        mean_session = float(sub['avg_session_mins'].mean())
        mean_num_sessions = float(sub['num_sessions'].mean())
        mean_diversity = float(sub['genre_diversity'].mean())
        mean_weekend = float(sub['weekend_watch_ratio'].mean())
        mean_recency = float(sub['days_since_last_active'].mean())

        dominant_genres = [g for g in ALL_GENRES if sub[f'genre_{g}'].mean() >= 0.30]

        if mean_diversity >= 3.5:
            name = 'Eclectic Multi-Genre Explorers'
        elif mean_weekend >= 0.70:
            name = 'Weekend Binge Enthusiasts'
        elif mean_watch >= 45.0 and ('Action' in dominant_genres or 'Thriller' in dominant_genres):
            name = 'High-Engagement Action Viewers'
        elif mean_recency >= 20.0 or mean_watch < 6.0:
            name = 'Low-Activity Dormant Viewers'
        elif mean_session <= 35.0:
            name = 'Casual Short-Session Viewers'
        else:
            name = f'Mainstream Viewer Cohort {c}'

        segment_names[c] = name
        profiles[c] = {
            'cluster_id': c,
            'segment_name': name,
            'user_count': count,
            'user_percentage': round(pct * 100, 2),
            'mean_watch_time_hours': round(mean_watch, 2),
            'mean_avg_session_mins': round(mean_session, 1),
            'mean_num_sessions': round(mean_num_sessions, 1),
            'mean_genre_diversity': round(mean_diversity, 2),
            'mean_weekend_ratio': round(mean_weekend, 2),
            'mean_recency_days': round(mean_recency, 1),
            'dominant_genres': dominant_genres
        }
        logger.info(f"Segment {c} assigned: '{name}' (n={count}, {pct:.1%})")

    return segment_names, profiles

def train_and_persist():
    logger.info('Starting Audience Segmentation training pipeline...')
    data_path = find_dataset_path()
    raw_df = pd.read_csv(data_path)
    clean_df, defaults = clean_and_prepare_data(raw_df)

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(clean_df[NUM_COLS])
    X_genre = clean_df[GENRE_COLS].values
    X_full = np.hstack([X_num_scaled, X_genre])

    best_k, k_comparison = evaluate_cluster_counts(X_full, min_k=2, max_k=6)

    logger.info(f'Fitting final KMeans model with K={best_k}, random_state=42...')
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(X_full)

    final_silhouette = float(silhouette_score(X_full, final_labels))
    final_inertia = float(kmeans.inertia_)
    final_db = float(davies_bouldin_score(X_full, final_labels))
    final_ch = float(calinski_harabasz_score(X_full, final_labels))
    cluster_counts = {int(c): int(n) for c, n in pd.Series(final_labels).value_counts().items()}

    segment_names, cluster_profiles = generate_segment_names(clean_df, final_labels, best_k)

    model_bundle = {
        'version': '1.0.0',
        'algorithm': 'KMeans',
        'random_state': 42,
        'n_clusters': best_k,
        'scaler': scaler,
        'kmeans': kmeans,
        'num_cols': NUM_COLS,
        'genre_cols': GENRE_COLS,
        'all_genres': ALL_GENRES,
        'defaults': defaults,
        'cluster_profiles': cluster_profiles,
        'segment_names': segment_names,
        'recommendation_catalog': RECOMMENDATION_CATALOG,
        'evaluation_metrics': {
            'silhouette_score': round(final_silhouette, 4),
            'inertia': round(final_inertia, 2),
            'davies_bouldin': round(final_db, 4),
            'calinski_harabasz': round(final_ch, 2),
            'cluster_sizes': cluster_counts
        },
        'k_comparison': k_comparison
    }

    models_dir = os.environ.get('MODELS_DIR', 'models')
    os.makedirs(models_dir, exist_ok=True)
    bundle_path = os.path.join(models_dir, 'model_bundle.joblib')
    joblib.dump(model_bundle, bundle_path)
    logger.info(f'Model artifact successfully persisted to {bundle_path}')

    metrics_summary_path = os.path.join(models_dir, 'training_metrics.json')
    with open(metrics_summary_path, 'w') as f:
        json.dump(model_bundle['evaluation_metrics'], f, indent=2)

    logger.info('Training pipeline completed successfully.')
    return bundle_path

if __name__ == '__main__':
    train_and_persist()
