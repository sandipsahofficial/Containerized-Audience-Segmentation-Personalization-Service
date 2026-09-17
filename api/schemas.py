from typing import Dict, Any, Tuple, Optional, List

def validate_recommend_payload(data: Any) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates the incoming recommendation request payload.
    Returns: (is_valid, error_message, sanitized_data)
    """
    if not isinstance(data, dict):
        return False, "Request payload must be a valid JSON object.", None

    # 1. Validate user_id
    if 'user_id' not in data:
        return False, "Missing required field: 'user_id'.", None
    user_id = data['user_id']
    if not isinstance(user_id, (str, int)) or not str(user_id).strip():
        return False, "Field 'user_id' must be a non-empty string or integer identifier.", None
    user_id_clean = str(user_id).strip()

    # 2. Validate watch_time_hours
    if 'watch_time_hours' not in data:
        return False, "Missing required field: 'watch_time_hours'.", None
    watch_time = data['watch_time_hours']
    if isinstance(watch_time, bool) or not isinstance(watch_time, (int, float)):
        return False, "Field 'watch_time_hours' must be a valid numeric value.", None
    if watch_time < 0:
        return False, "Field 'watch_time_hours' cannot be negative.", None
    if watch_time > 1000:
        return False, "Field 'watch_time_hours' exceeds maximum allowable threshold (1000 hours).", None

    # 3. Validate avg_session_mins
    if 'avg_session_mins' not in data:
        return False, "Missing required field: 'avg_session_mins'.", None
    avg_session = data['avg_session_mins']
    if isinstance(avg_session, bool) or not isinstance(avg_session, (int, float)):
        return False, "Field 'avg_session_mins' must be a valid numeric value.", None
    if avg_session <= 0:
        return False, "Field 'avg_session_mins' must be greater than zero.", None
    if avg_session > 600:
        return False, "Field 'avg_session_mins' exceeds maximum allowable threshold (600 minutes).", None

    # 4. Validate top_genres
    genres_raw = data.get('top_genres', [])
    top_genres_clean: List[str] = []

    if isinstance(genres_raw, list):
        for item in genres_raw:
            if isinstance(item, str) and item.strip():
                top_genres_clean.append(item.strip())
    elif isinstance(genres_raw, str):
        parts = [p.strip() for p in genres_raw.split(',') if p.strip()]
        top_genres_clean.extend(parts)
    elif genres_raw is None:
        top_genres_clean = []
    else:
        return False, "Field 'top_genres' must be a list of genre names or a comma-separated string.", None

    # 5. Optional fields validation
    num_sessions = data.get('num_sessions')
    if num_sessions is not None:
        if isinstance(num_sessions, bool) or not isinstance(num_sessions, (int, float)) or num_sessions < 0:
            return False, "Field 'num_sessions' must be a non-negative numeric value.", None
        num_sessions = int(num_sessions)

    weekend_ratio = data.get('weekend_watch_ratio')
    if weekend_ratio is not None:
        if isinstance(weekend_ratio, bool) or not isinstance(weekend_ratio, (int, float)) or not (0.0 <= weekend_ratio <= 1.0):
            return False, "Field 'weekend_watch_ratio' must be a float between 0.0 and 1.0.", None
        weekend_ratio = float(weekend_ratio)

    recency = data.get('days_since_last_active')
    if recency is not None:
        if isinstance(recency, bool) or not isinstance(recency, (int, float)) or recency < 0:
            return False, "Field 'days_since_last_active' must be a non-negative integer.", None
        recency = int(recency)

    sanitized = {
        'user_id': user_id_clean,
        'watch_time_hours': float(watch_time),
        'avg_session_mins': float(avg_session),
        'top_genres': top_genres_clean,
        'num_sessions': num_sessions,
        'weekend_watch_ratio': weekend_ratio,
        'days_since_last_active': recency
    }

    return True, None, sanitized
