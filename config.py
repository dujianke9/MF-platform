"""全局配置"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL",   "deepseek-chat")

# 权重计算平滑因子
SMOOTH = 1.0

# 临时文件目录
PLATFORM_TMP_DIR = os.getenv('PLATFORM_TMP_DIR', '/tmp/formula_platform')
# MinerU 的 conda 环境名称
MINERU_CONDA_ENV = os.getenv('MINERU_CONDA_ENV', 'mineru')
# MinerU 模型源
MINERU_MODEL_SOURCE = os.getenv('MINERU_MODEL_SOURCE', 'huggingface')