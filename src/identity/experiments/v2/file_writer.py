import base64
import sys
from pathlib import Path

target_file = Path(sys.argv[1])
content = base64.b64decode(sys.argv[2]).decode("utf-8")
target_file.parent.mkdir(parents=True, exist_ok=True)
target_file.write_text(content, encoding="utf-8")
print(f"Wrote {target_file} ({len(content)} chars)")
