from typing import List, Dict, Any

def get_recommendations(segment_name: str, top_genres: List[str], catalog: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
    """
    Transparent rule-based recommendation engine.
    Applies segment-specific strategy combined with content metadata and popularity.
    """
    user_genres_lower = [g.lower() for g in top_genres]

    def has_matching_genre(item):
        item_genres_lower = [g.lower() for g in item.get('genres', [])]
        return any(g in item_genres_lower for g in user_genres_lower)

    # 1. Segment-Specific Filtering Strategy
    if "High-Engagement Action" in segment_name:
        # Prioritize Action/Thriller content with strong ratings
        candidates = [
            item for item in catalog 
            if any(g in item.get('genres', []) for g in ['Action', 'Thriller'])
        ]
    elif "Casual Short-Session" in segment_name:
        # Prioritize short/bite-sized content (<= 30 mins) or Comedy/Animation
        candidates = [
            item for item in catalog 
            if item.get('duration_mins', 100) <= 30 or any(g in item.get('genres', []) for g in ['Comedy', 'Animation'])
        ]
    elif "Weekend Binge" in segment_name:
        # Prioritize immersive Series or long feature films in Drama/Sci-Fi
        candidates = [
            item for item in catalog 
            if item.get('type') == 'Series' or item.get('duration_mins', 0) >= 110
        ]
    elif "Eclectic Multi-Genre" in segment_name:
        # Match user's diverse preferred genres + adjacent discovery
        candidates = [
            item for item in catalog 
            if has_matching_genre(item) or item.get('popularity', 0) >= 90
        ]
    elif "Low-Activity" in segment_name:
        # Prioritize top-popularity, low-friction, widely loved content
        candidates = sorted(catalog, key=lambda x: x.get('popularity', 0), reverse=True)
    else:
        candidates = list(catalog)

    # Fallback if filtered list is too small
    if len(candidates) < top_k:
        candidates = catalog

    # 2. Ranking Strategy:
    # Score items based on:
    # +20 if item shares an explicit genre with user's top_genres
    # +popularity / 5
    def rank_score(item):
        score = item.get('popularity', 50) / 5.0
        if has_matching_genre(item):
            score += 20.0
        return score

    sorted_candidates = sorted(candidates, key=rank_score, reverse=True)

    # Return top_k unique titles
    recommended_titles = []
    seen = set()
    for item in sorted_candidates:
        title = item['title']
        if title not in seen:
            seen.add(title)
            recommended_titles.append(title)
            if len(recommended_titles) >= top_k:
                break

    return recommended_titles
