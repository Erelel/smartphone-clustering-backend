import json
import os
import numpy as np
from firebase_functions import https_fn
from firebase_admin import initialize_app, firestore, credentials

# ----------------- 수정된 부분 시작 -----------------
# 서비스 계정 키 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(current_dir, 'serviceAccountKey.json')

# 로컬에 키 파일이 존재하면 키를 사용하여 초기화 (로컬 테스트용)
if os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
    initialize_app(cred)
# 키 파일이 없으면 기본 인증으로 초기화 (클라우드 배포용)
else:
    initialize_app()

db = firestore.client()
# ----------------- 수정된 부분 끝 -----------------
# 모델 파라미터 로드 함수
def load_model_params():
    # 현재 파일(main.py)과 같은 디렉토리의 json 파일 읽기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'model_params.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 중심점과의 거리 계산 함수
def calculate_distances(features, centroids):
    # np.linalg.norm(a - b)를 사용하여 유클리디안 거리 연산
    distances = [np.linalg.norm(np.array(features) - np.array(c)) for c in centroids]
    cluster_id = int(np.argmin(distances))
    min_distance = float(distances[cluster_id])
    return cluster_id, min_distance

# 수동 MinMax 스케일링 함수
def scale_features(focus, switch, params):
    # (x - min) / (max - min) 공식 적용
    scaled_focus = (focus - params['focus_min']) / (params['focus_max'] - params['focus_min'])
    scaled_switch = (switch - params['switch_min']) / (params['switch_max'] - params['switch_min'])
    return [scaled_focus, scaled_switch]

@https_fn.on_request()
def predict_user_cluster(req: https_fn.Request) -> https_fn.Response:
    """앱에서 호출하는 API 엔드포인트"""
    try:
        # 1. 요청 데이터 파싱
        req_data = req.get_json()
        user_id = req_data.get('user_id')
        duration_sum = req_data.get('foreground_app_duration_sum')
        switch_per_hour = req_data.get('foreground_app_switch_per_hour')
        concentration_ratio = req_data.get('concentration_ratio')
        week_start = req_data.get('week_start')
        
        if not all([user_id, duration_sum, switch_per_hour, concentration_ratio]):
            return https_fn.Response("Missing required fields", status=400)

        # 2. 파라미터 로드
        params = load_model_params()

        # 3. 파생 변수 연산 (preprocessing.py 로직 동일 적용)
        # log1p 적용 여부 체크 후 연산
        if params.get('log_transform_applied', False):
            duration_sum = np.log1p(duration_sum)
            switch_per_hour = np.log1p(switch_per_hour)
            
        focus_intensity = duration_sum * concentration_ratio
        switch_frequency = switch_per_hour * (1 - concentration_ratio)

        # 4. 스케일링 및 군집 판별
        scaled_features = scale_features(focus_intensity, switch_frequency, params)
        cluster_id, distance = calculate_distances(scaled_features, params['centroids'])

        # 5. 결과 데이터 구성
        result_data = {
            "week_start": week_start,
            "features": {
                "focus_intensity": float(focus_intensity),
                "switch_frequency": float(switch_frequency)
            },
            "clustering_result": {
                "assigned_cluster_id": cluster_id,
                "distance_to_centroid": distance
            },
            "created_at": firestore.SERVER_TIMESTAMP
        }

        # 6. Firestore에 저장 (권장 스키마: users/{user_id}/weekly_clusters/{week_start})
        doc_ref = db.collection('users').document(user_id).collection('weekly_clusters').document(week_start)
        doc_ref.set(result_data)

        # 7. 클라이언트로 결과 반환
        return https_fn.Response(json.dumps(result_data), status=200, mimetype="application/json")

    except Exception as e:
        return https_fn.Response(f"Error processing request: {str(e)}", status=500)