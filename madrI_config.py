# madrI_config.py
import os

# Data paths - sesuaikan jika perlu
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

FLOOD_SHP_PATH = os.path.join(DATA_DIR, "Genangan_Revisi_Lagi.shp")
EVAC_GEOJSON_PATH = os.path.join(
    DATA_DIR, "titik_evakuasi_2_dengan_alamat baru.geojson"
)
USER_SHP_PATH = os.path.join(DATA_DIR, "PEMUKIMAN_AR_25K.shp")
ELEVATION_TIF_PATH = os.path.join(DATA_DIR, "output_hh.tif")

# MADRL hyperparams
NUM_AGENTS = 16  # Agen tetap per episode (A3 pilihanmu)
NUM_EPISODES = 200
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.99

# Filtering radius (km) saat cari kandidat per agen (one-shot local focus)
CANDIDATE_RADIUS_KM = 10.0

# Congestion penalty param (R2: density-based)
CONGESTION_ALPHA = 3.0

# Device
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OSRM
BASE_URL_OSRM = "http://localhost:3006"  # adapt sesuai env kamu
