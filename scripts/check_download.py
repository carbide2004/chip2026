import time
import os
from pathlib import Path

target_dir = Path(r"C:\Users\HITSZ\.cache\modelscope\models\Qwen--Qwen3-8B\snapshots\master")
total_expected_gb = 15.26

print(f"Monitoring Qwen3-8B download progress (Target: {total_expected_gb:.2f} GB)...")
last_bytes = 0

for _ in range(5):
    if target_dir.exists():
        files = list(target_dir.glob("*.safetensors*"))
        cur_bytes = sum(f.stat().st_size for f in files)
        cur_gb = cur_bytes / (1024**3)
        pct = (cur_gb / total_expected_gb) * 100
        speed_mb = ((cur_bytes - last_bytes) / (1024**2)) / 3 if last_bytes > 0 else 0
        last_bytes = cur_bytes
        print(f"Progress: {cur_gb:.2f} / {total_expected_gb:.2f} GB ({pct:.1f}%), Files: {len(files)}, Speed: ~{speed_mb:.2f} MB/s")
    else:
        print("Waiting for directory creation...")
    time.sleep(3)
