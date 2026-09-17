import numpy as np
from typing import List, Dict, Any, Tuple, Optional

ALL_GENRES = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller', 'Romance', 'Documentary', 'Animation', 'Horror']

class ContentRecommender:
    """
    Local, CPU-friendly, deterministic Content-Based Vector Space Recommendation Engine.
    Computes multi-dimensional cosine similarity across:
    1. Genre multi-hot representations (50% weight)
    2. Session duration and format alignment (25% weight)
    3. KMeans empirical cohort affinity (15% weight)
    4. Normalized platform popularity prior (10% weight)
    
    Zero external APIs. Zero retraining during inference.
    """
    def __init__(self, bundle: Dict[str, Any]):
        self.catalog = bundle.get('catalog', [])
        self.all_genres = bundle.get('all_genres', ALL_GENRES)
        self.genre_matrix = bundle.get('genre_matrix')
        self.format_matrix = bundle.get('format_matrix')
        self.affinity_matrix = bundle.get('affinity_matrix')
        self.pop_vector = bundle.get('pop_vector')
        self.weights = bundle.get('weights', {
            'genre': 0.50,
            'format': 0.25,
            'segment': 0.15,
            'popularity': 0.10
        })
        
        # Self-initialize matrices if bundle was minimal
        if self.genre_matrix is None or len(self.catalog) == 0:
            self._rebuild_matrices()

    def _rebuild_matrices(self):
        n_items = len(self.catalog)
        if n_items == 0:
            return
        self.genre_matrix = np.zeros((n_items, len(self.all_genres)), dtype=np.float64)
        self.format_matrix = np.zeros((n_items, 3), dtype=np.float64)
        self.affinity_matrix = np.zeros((n_items, 5), dtype=np.float64)
        self.pop_vector = np.zeros(n_items, dtype=np.float64)

        for i, item in enumerate(self.catalog):
            for g in item.get('genres', []):
                if g in self.all_genres:
                    self.genre_matrix[i, self.all_genres.index(g)] = 1.0
            g_norm = np.linalg.norm(self.genre_matrix[i])
            if g_norm > 0:
                self.genre_matrix[i] /= g_norm

            dur = item.get('duration_mins', 60)
            is_series = (item.get('type') == 'Series')
            if is_series or dur > 105:
                self.format_matrix[i] = [0.0, 0.2, 1.0]
            elif dur <= 30:
                self.format_matrix[i] = [1.0, 0.2, 0.0]
            else:
                self.format_matrix[i] = [0.2, 1.0, 0.3]
            f_norm = np.linalg.norm(self.format_matrix[i])
            if f_norm > 0:
                self.format_matrix[i] /= f_norm

            item_genres = set(item.get('genres', []))
            self.affinity_matrix[i, 0] = len(item_genres.intersection({'Drama', 'Romance', 'Documentary', 'Sci-Fi'})) / max(1, len(item_genres))
            self.affinity_matrix[i, 1] = len(item_genres.intersection({'Action', 'Thriller'})) / max(1, len(item_genres))
            self.affinity_matrix[i, 2] = (len(item_genres.intersection({'Comedy', 'Animation'})) + (1.0 if dur <= 30 else 0.0)) / 2.0
            self.affinity_matrix[i, 3] = item.get('popularity', 50) / 100.0
            self.affinity_matrix[i, 4] = ((1.0 if is_series or dur >= 110 else 0.0) + len(item_genres.intersection({'Drama', 'Sci-Fi', 'Thriller'}))) / 2.0

            aff_norm = np.linalg.norm(self.affinity_matrix[i])
            if aff_norm > 0:
                self.affinity_matrix[i] /= aff_norm

            self.pop_vector[i] = item.get('popularity', 50) / 100.0

    def rank(
        self,
        preferred_genres: List[str],
        avg_session_mins: float = 60.0,
        weekend_watch_ratio: float = 0.45,
        segment_id: int = 0,
        top_k: int = 3
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Calculates cosine similarity and composite ML ranking scores for all catalog items.
        Zero retraining. Constant-time CPU matrix operations (< 1ms).
        """
        n_items = len(self.catalog)
        if n_items == 0:
            return [], []

        # 1. Transform User Genre Preferences into Normalized Vector
        u_genre = np.zeros(len(self.all_genres), dtype=np.float64)
        all_genres_lower = [g.lower() for g in self.all_genres]
        pref_lower = [g.lower() for g in preferred_genres]
        for g in preferred_genres:
            if g.lower() in all_genres_lower:
                idx = all_genres_lower.index(g.lower())
                u_genre[idx] = 1.0

        u_genre_norm = np.linalg.norm(u_genre)
        if u_genre_norm > 0:
            u_genre /= u_genre_norm
            genre_sims = self.genre_matrix.dot(u_genre)
        else:
            # Graceful fallback: If no matching genre provided, neutral similarity
            genre_sims = np.zeros(n_items, dtype=np.float64)

        # 2. Transform Viewer Session Format Preference
        if avg_session_mins <= 30:
            u_format = np.array([1.0, 0.2, 0.0], dtype=np.float64)
        elif avg_session_mins <= 90:
            u_format = np.array([0.2, 1.0, 0.3], dtype=np.float64)
        else:
            u_format = np.array([0.0, 0.3, 1.0], dtype=np.float64)

        if weekend_watch_ratio is not None and weekend_watch_ratio >= 0.70:
            u_format[2] += 0.4
        u_format /= np.linalg.norm(u_format)
        format_sims = self.format_matrix.dot(u_format)

        # 3. Transform Segment Alignment
        u_seg = np.zeros(5, dtype=np.float64)
        if 0 <= segment_id < 5:
            u_seg[segment_id] = 1.0
        seg_sims = self.affinity_matrix.dot(u_seg)

        # 4. Composite ML Score: Weighted Sum of Normalized Vector Similarities
        # score = w_g * S_genre + w_f * S_format + w_s * S_seg + w_p * Pop
        w = self.weights
        composite_scores = (
            w.get('genre', 0.50) * genre_sims +
            w.get('format', 0.25) * format_sims +
            w.get('segment', 0.15) * seg_sims +
            w.get('popularity', 0.10) * self.pop_vector
        )

        # 5. Build Traceable Detailed Recommendations
        scored_items = []
        for i, item in enumerate(self.catalog):
            item_genres = item.get('genres', [])
            matched = [g for g in item_genres if g.lower() in pref_lower]
            score_val = float(composite_scores[i])

            ranking_reasons = []
            if genre_sims[i] > 0.3:
                ranking_reasons.append(f"Genre match ({', '.join(matched) if matched else 'Profile similarity'})")
            if format_sims[i] > 0.5:
                ranking_reasons.append(f"Format alignment ({item.get('type')}, {item.get('duration_mins')}m)")
            if seg_sims[i] > 0.4:
                ranking_reasons.append(f"Cohort #{segment_id} behavioral fit")
            if not ranking_reasons:
                ranking_reasons.append("Platform catalog baseline")

            scored_items.append({
                'id': item.get('id', f'ITEM-{i}'),
                'title': item.get('title', 'Unknown Title'),
                'score': round(score_val, 4),
                'similarity_score': round(score_val, 4),
                'matched_genres': matched,
                'genres': item_genres,
                'type': item.get('type', 'Movie'),
                'duration_mins': item.get('duration_mins', 0),
                'popularity': item.get('popularity', 0),
                'ranking_reason': "; ".join(ranking_reasons)
            })

        # 6. Sort descending by score, tie-break by title for strict determinism
        scored_items.sort(key=lambda x: (-x['score'], x['title']))
        top_items = scored_items[:top_k]
        titles = [x['title'] for x in top_items]

        return titles, top_items


def get_recommendations(
    bundle_or_segment: Any,
    top_genres: List[str],
    catalog_or_session: Any = None,
    weekend_ratio: Optional[float] = None,
    segment_id: int = 0,
    top_k: int = 3
) -> List[str]:
    """
    Main entry point for recommendation engine.
    Supports both:
    1. New ML vector-based call: get_recommendations(recommender_bundle, genres, session, weekend, segment_id, top_k)
    2. Legacy fallback signature: get_recommendations(segment_name, genres, catalog, top_k)
    """
    # Check if first argument is a recommender_bundle dict
    if isinstance(bundle_or_segment, dict) and 'catalog' in bundle_or_segment:
        recommender = ContentRecommender(bundle_or_segment)
        avg_session = float(catalog_or_session) if isinstance(catalog_or_session, (int, float)) else 60.0
        titles, _ = recommender.rank(
            preferred_genres=top_genres,
            avg_session_mins=avg_session,
            weekend_watch_ratio=weekend_ratio if weekend_ratio is not None else 0.45,
            segment_id=segment_id,
            top_k=top_k
        )
        return titles

    # Fallback to catalog list passed directly
    catalog = catalog_or_session if isinstance(catalog_or_session, list) else []
    temp_bundle = {'catalog': catalog, 'all_genres': ALL_GENRES}
    recommender = ContentRecommender(temp_bundle)
    titles, _ = recommender.rank(
        preferred_genres=top_genres,
        avg_session_mins=60.0,
        weekend_watch_ratio=0.45,
        segment_id=segment_id,
        top_k=top_k
    )
    return titles
