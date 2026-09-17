import sys
from pathlib import Path
import subprocess

repo_root = Path(__file__).resolve().parent.parent

def run_cmd_streaming(cmd, log_file=None):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    with open(log_file, "a", encoding="utf-8") if log_file else None as f:
        for line in process.stdout:
            print(line, end="", flush=True)
            if f:
                f.write(line)
                f.flush()
    process.wait()
    return process.returncode
