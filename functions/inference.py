import math

import numpy as np


def _restore_centroids_from_storage(centroids) -> list[list[float]]:
    if isinstance(centroids, dict):
        ordered_items = sorted(centroids.items(), key=lambda item: int(item[0]))
        return [[float(value) for value in row] for _, row in ordered_items]

    if isinstance(centroids, list):
        return [[float(value) for value in row] for row in centroids]

    raise ValueError("Model params centroids are missing or invalid.")


def _normalize_model_params(params: dict) -> dict:
    if not isinstance(params, dict):
        raise ValueError("Model params must be a dictionary.")

    normalized = dict(params)
    normalized["centroids"] = _restore_centroids_from_storage(
        normalized.get("centroids")
    )
    return normalized


def _derive_features(payload: dict, use_log_transform: bool) -> dict:
    duration_base = float(payload["foreground_app_duration_sum"])
    switch_base = float(payload["foreground_app_switch_per_hour"])
    concentration_ratio = float(payload["concentration_ratio"])

    if use_log_transform:
        duration_base = math.log1p(duration_base)
        switch_base = math.log1p(switch_base)

    focus_intensity = duration_base * concentration_ratio
    switch_frequency = switch_base * (1.0 - concentration_ratio)

    return {
        "focus_intensity": float(focus_intensity),
        "switch_frequency": float(switch_frequency),
    }


def _minmax_scale(value: float, data_min: float, data_max: float) -> float:
    denom = data_max - data_min
    if denom == 0:
        return 0.0
    return (value - data_min) / denom


def infer_cluster(payload: dict, params: dict) -> dict:
    params = _normalize_model_params(params)

    features = _derive_features(payload, params["log_transform_applied"])
    data_min = params["scaler"]["data_min"]
    data_max = params["scaler"]["data_max"]

    scaled_focus = _minmax_scale(features["focus_intensity"], data_min[0], data_max[0])
    scaled_switch = _minmax_scale(
        features["switch_frequency"], data_min[1], data_max[1]
    )

    vector = np.array([scaled_focus, scaled_switch], dtype=float)
    centroids = np.asarray(params["centroids"], dtype=float)
    distances = np.linalg.norm(centroids - vector, axis=1)
    assigned_cluster_id = int(np.argmin(distances))
    distance_to_centroid = float(distances[assigned_cluster_id])

    return {
        "week_start": payload["week_start"],
        "week_end": payload["week_end"],
        "is_valid_data": payload["is_valid_data"],
        "features": features,
        "clustering_result": {
            "assigned_cluster_id": assigned_cluster_id,
            "distance_to_centroid": distance_to_centroid,
        },
    }
