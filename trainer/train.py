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
    {'id': 'MOV-01', 'title': 'Inception', 'genres': ['Action', 'Sci-Fi'], 'type': 'Movie', 'duration_mins': 148, 'popularity': 98},
    {'id': 'MOV-02', 'title': 'The Dark Knight', 'genres': ['Action', 'Thriller'], 'type': 'Movie', 'duration_mins': 152, 'popularity': 99},
    {'id': 'MOV-03', 'title': 'Gladiator', 'genres': ['Action', 'Drama'], 'type': 'Movie', 'duration_mins': 155, 'popularity': 93},
    {'id': 'MOV-04', 'title': 'Interstellar', 'genres': ['Sci-Fi', 'Drama'], 'type': 'Movie', 'duration_mins': 169, 'popularity': 97},
    {'id': 'SER-01', 'title': 'Breaking Bad', 'genres': ['Drama', 'Thriller'], 'type': 'Series', 'duration_mins': 49, 'popularity': 99},
    {'id': 'SER-02', 'title': 'The Crown', 'genres': ['Drama', 'Thriller'], 'type': 'Series', 'duration_mins': 55, 'popularity': 92},
    {'id': 'SER-03', 'title': 'Stranger Things', 'genres': ['Sci-Fi', 'Drama', 'Horror'], 'type': 'Series', 'duration_mins': 52, 'popularity': 96},
    {'id': 'SHT-01', 'title': 'The Office', 'genres': ['Comedy'], 'type': 'Short', 'duration_mins': 22, 'popularity': 95},
    {'id': 'SHT-02', 'title': 'Rick and Morty', 'genres': ['Comedy', 'Animation'], 'type': 'Short', 'duration_mins': 23, 'popularity': 94},
    {'id': 'MOV-05', 'title': 'La La Land', 'genres': ['Romance', 'Comedy'], 'type': 'Movie', 'duration_mins': 128, 'popularity': 91},
    {'id': 'DOC-01', 'title': 'Our Planet', 'genres': ['Documentary'], 'type': 'Series', 'duration_mins': 50, 'popularity': 94},
    {'id': 'DOC-02', 'title': 'The Social Dilemma', 'genres': ['Documentary', 'Sci-Fi'], 'type': 'Movie', 'duration_mins': 94, 'popularity': 90},
    {'id': 'MOV-06', 'title': 'A Quiet Place', 'genres': ['Horror', 'Thriller'], 'type': 'Movie', 'duration_mins': 90, 'popularity': 89},
    {'id': 'SHT-03', 'title': 'Arcane', 'genres': ['Animation', 'Action'], 'type': 'Short', 'duration_mins': 40, 'popularity': 97},
    {'id': 'MOV-07', 'title': 'Money Heist', 'genres': ['Action', 'Thriller'], 'type': 'Series', 'duration_mins': 50, 'popularity': 96},
]

def load_catalog():
    candidate_paths = [
        os.environ.get('CATALOG_PATH'),
        'data/ott_catalog.csv',
        '/app/data/ott_catalog.csv',
        '../data/ott_catalog.csv',
    ]
    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                cat_df = pd.read_csv(p)
                catalog = []
                for _, row in cat_df.iterrows():
                    genres = [g.strip() for g in str(row['genres']).split(',')]
                    catalog.append({
                        'id': str(row['id']),
                        'title': str(row['title']),
                        'genres': genres,
                        'type': str(row['type']),
                        'duration_mins': int(row['duration_mins']),
                        'popularity': int(row.get('popularity', 90)),
                        'imdb_rating': float(row.get('imdb_rating', 8.5)),
                        'platform': str(row.get('platform', 'OTT Platform'))
                    })
                logger.info(f"Loaded {len(catalog)} real titles from {p}")
                return catalog
            except Exception as e:
                logger.warning(f"Failed loading {p}: {e}, using default catalog.")
    return RECOMMENDATION_CATALOG

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

def build_recommendation_matrices(catalog, all_genres):
    """
    Precomputes normalized vector representations for content-based ML recommendation:
    1. Genre Multi-Hot Matrix (L2 normalized)
    2. Format / Session Duration Matrix (L2 normalized)
    3. Cohort Empirical Affinity Matrix (L2 normalized)
    4. Popularity Prior Vector
    """
    n_items = len(catalog)
    genre_matrix = np.zeros((n_items, len(all_genres)), dtype=np.float64)
    format_matrix = np.zeros((n_items, 3), dtype=np.float64)
    affinity_matrix = np.zeros((n_items, 5), dtype=np.float64)
    pop_vector = np.zeros(n_items, dtype=np.float64)

    for i, item in enumerate(catalog):
        # 1. Multi-hot genre vector
        for g in item.get('genres', []):
            if g in all_genres:
                genre_matrix[i, all_genres.index(g)] = 1.0
        g_norm = np.linalg.norm(genre_matrix[i])
        if g_norm > 0:
            genre_matrix[i] /= g_norm

        # 2. Format / duration vector
        dur = item.get('duration_mins', 60)
        is_series = (item.get('type') == 'Series')
        if is_series or dur > 105:
            format_matrix[i] = [0.0, 0.2, 1.0]
        elif dur <= 30:
            format_matrix[i] = [1.0, 0.2, 0.0]
        else:
            format_matrix[i] = [0.2, 1.0, 0.3]
        f_norm = np.linalg.norm(format_matrix[i])
        if f_norm > 0:
            format_matrix[i] /= f_norm

        # 3. Empirical cohort affinity
        item_genres = set(item.get('genres', []))
        affinity_matrix[i, 0] = len(item_genres.intersection({'Drama', 'Romance', 'Documentary', 'Sci-Fi'})) / max(1, len(item_genres))
        affinity_matrix[i, 1] = len(item_genres.intersection({'Action', 'Thriller'})) / max(1, len(item_genres))
        affinity_matrix[i, 2] = (len(item_genres.intersection({'Comedy', 'Animation'})) + (1.0 if dur <= 30 else 0.0)) / 2.0
        affinity_matrix[i, 3] = item.get('popularity', 50) / 100.0
        affinity_matrix[i, 4] = ((1.0 if is_series or dur >= 110 else 0.0) + len(item_genres.intersection({'Drama', 'Sci-Fi', 'Thriller'}))) / 2.0

        aff_norm = np.linalg.norm(affinity_matrix[i])
        if aff_norm > 0:
            affinity_matrix[i] /= aff_norm

        # 4. Normalized popularity prior
        pop_vector[i] = item.get('popularity', 50) / 100.0

    return {
        'genre_matrix': genre_matrix,
        'format_matrix': format_matrix,
        'affinity_matrix': affinity_matrix,
        'pop_vector': pop_vector,
        'catalog': catalog,
        'all_genres': all_genres,
        'weights': {
            'genre': 0.50,
            'format': 0.25,
            'segment': 0.15,
            'popularity': 0.10
        },
        'method': 'Content-Based Vector Space & Multi-Feature Cosine Similarity'
    }

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

    # Build and persist content-based ML recommendation model vectors
    logger.info('Building content-based vector space representations for recommendation catalog...')
    catalog = load_catalog()
    recommender_bundle = build_recommendation_matrices(catalog, ALL_GENRES)

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
        'recommendation_catalog': catalog,
        'recommender_bundle': recommender_bundle,
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
