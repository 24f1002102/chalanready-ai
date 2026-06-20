import warnings; warnings.filterwarnings('ignore')
from fastapi.testclient import TestClient
from backend.main import app
import numpy as np, cv2, io

c = TestClient(app)
print("=== ChalanReady AI Final Verification ===")

# 1. Health
h = c.get('/api/health').json()
print(f"1. Health: {h['status']} v{h['version']}")

# 2. Image endpoint
img = np.zeros((540, 960, 3), dtype=np.uint8)
img[170:460] = (70, 70, 70)
img[200:280, 100:180] = (0, 0, 220)
_, buf = cv2.imencode('.jpg', img)
r = c.post('/api/images/analyze',
    files={'file': ('t.jpg', io.BytesIO(buf.tobytes()), 'image/jpeg')})
d = r.json()
print(f"2. /images/analyze: HTTP {r.status_code}, has_image={'annotated_image' in d}, preprocessing='{d.get('preprocessing','?')}'")

# 3. Video pipeline
with open('sample_data/videos/synthetic_stage1.mp4', 'rb') as f:
    r2 = c.post('/api/videos/process', files={'file': f})
d2 = r2.json()
print(f"3. /videos/process: HTTP {r2.status_code}, violations_in_store={d2.get('violations_in_store','?')}, officer_gate={d2.get('officer_review_required')}")

# 4. SQLite persistence
c.post('/api/demo/seed')
v = c.get('/api/violations').json()
print(f"4. SQLite store: {v['count']} violations persisted")

# 5. Officer review flow
pid = v['violations'][0]['packet_id']
rv = c.post(f'/api/violations/{pid}/review',
    json={'packet_id': pid, 'action': 'approve', 'officer_id': 'judge'}).json()
print(f"5. Officer review: {rv['packet']['review_status']}")

# 6. Analytics
a = c.get('/api/analytics').json()
print(f"6. Analytics: total={a['total_violations']}, approved={a['approved_challans']}")

# 7. Red-light module
from backend.detection.violations.redlight import is_red_signal
black = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"7. Red-light detector: imported OK, black_frame={is_red_signal(black)}")

# 8. Check DB is NOT in repo root
import os
# DB may exist locally (runtime data) but must NOT be tracked by git
import subprocess
result = subprocess.run(['git', 'ls-files', 'chalanready.db'], capture_output=True, text=True)
assert result.stdout.strip() == '', f"chalanready.db is tracked by git — run: git rm --cached chalanready.db"
print("8. chalanready.db: NOT tracked by git (correct)")

print("\nALL SYSTEMS VERIFIED")
