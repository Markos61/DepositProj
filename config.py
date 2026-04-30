from dotenv import load_dotenv
import os

load_dotenv()

name = os.environ.get("NAME")
inference_model_name = os.environ.get("INFERENCE_MODEL")