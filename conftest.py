"""pytest가 pip install 없이 src/ 패키지를 import하도록 경로 추가."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
