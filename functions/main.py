import json
import os
import threading
from typing import Any

from firebase_admin import credentials, firestore, initialize_app
from firebase_functions import https_fn

from inference import infer_cluster


current_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(current_dir, "serviceAccountKey.json")

if os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
    initialize_app(cred)
else:
    initialize_app()

db = firestore.client()

SYSTEM_PARAMS_COLLECTION = "system_parameters"
SYSTEM_PARAMS_DOC = "kmeans_v1"

_MODEL_PARAMS = None
_MODEL_PARAMS_LOCK = threading.Lock()


def _restore_centroids_from_firestore(centroids: Any) -> list[list[float]]:
    if isinstance(centroids, dict):
        ordered_items = sorted(centroids.items(), key=lambda item: int(item[0]))
        return [[float(value) for value in row] for _, row in ordered_items]

    if isinstance(centroids, list):
        return [[float(value) for value in row] for row in centroids]

    raise ValueError("Model params centroids are missing or invalid.")


def _load_model_params_from_firestore() -> dict:
    doc_ref = db.collection(SYSTEM_PARAMS_COLLECTION).document(SYSTEM_PARAMS_DOC)
    snapshot = doc_ref.get()
    if not snapshot.exists:
        raise ValueError("Model params document not found: system_parameters/kmeans_v1")

    params = snapshot.to_dict()
    if not isinstance(params, dict):
        raise ValueError("Model params document is empty or invalid.")

    params["centroids"] = _restore_centroids_from_firestore(params.get("centroids"))
    return params


def get_model_params() -> dict:
    global _MODEL_PARAMS
    if _MODEL_PARAMS is None:
        with _MODEL_PARAMS_LOCK:
            if _MODEL_PARAMS is None:
                _MODEL_PARAMS = _load_model_params_from_firestore()
    return _MODEL_PARAMS


def _parse_request(req: https_fn.Request) -> dict:
    req_data = req.get_json(silent=True)
    if not isinstance(req_data, dict):
        raise ValueError("Invalid JSON payload.")

    required_fields = [
        "user_id",
        "week_start",
        "week_end",
        "is_valid_data",
        "foreground_app_duration_sum",
        "foreground_app_switch_per_hour",
        "concentration_ratio",
    ]
    missing = [
        field
        for field in required_fields
        if field not in req_data or req_data[field] is None
    ]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return {
        "user_id": req_data["user_id"],
        "week_start": req_data["week_start"],
        "week_end": req_data["week_end"],
        "is_valid_data": req_data["is_valid_data"],
        "foreground_app_duration_sum": req_data["foreground_app_duration_sum"],
        "foreground_app_switch_per_hour": req_data["foreground_app_switch_per_hour"],
        "concentration_ratio": req_data["concentration_ratio"],
    }


@https_fn.on_request()
def predict_user_cluster(req: https_fn.Request) -> https_fn.Response:
    try:
        payload = _parse_request(req)
        params = get_model_params()

        inference_result = infer_cluster(payload, params)

        doc_data = {
            "user_id": payload["user_id"],
            "week_start": inference_result["week_start"],
            "week_end": inference_result["week_end"],
            "is_valid_data": bool(inference_result["is_valid_data"]),
            "focus_intensity": float(inference_result["features"]["focus_intensity"]),
            "switch_frequency": float(
                inference_result["features"]["switch_frequency"]
            ),
            "cluster_id": int(
                inference_result["clustering_result"]["assigned_cluster_id"]
            ),
            "distance_to_centroid": float(
                inference_result["clustering_result"]["distance_to_centroid"]
            ),
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        doc_ref = (
            db.collection("users")
            .document(payload["user_id"])
            .collection("weekly_clusters")
            .document(payload["week_start"])
        )
        doc_ref.set(doc_data)

        response_data = {k: v for k, v in doc_data.items() if k != "created_at"}
        return https_fn.Response(
            json.dumps(response_data, ensure_ascii=True),
            status=200,
            mimetype="application/json",
        )
    except ValueError as exc:
        return https_fn.Response(str(exc), status=400)
    except Exception as exc:
        return https_fn.Response(f"Error processing request: {str(exc)}", status=500)
