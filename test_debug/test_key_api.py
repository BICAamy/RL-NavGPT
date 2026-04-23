import os
import sys
from dotenv import load_dotenv

print("python解释器路径：", sys.executable)

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_BASE"))