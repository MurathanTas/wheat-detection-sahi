"""
download_dataset.py — Roboflow Universe'ten Kaggle Wheat-Head dataset'ini
indirir. API key .env dosyasından okunur.
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY bulunamadı. "
        ".env dosyasında tanımlandığından emin ol."
    )

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("kaggle-lfmsn").project("wheat-head-nwevg")
version = project.version(1)
dataset = version.download("yolov8")

print(f"\nDataset indirildi: {dataset.location}")