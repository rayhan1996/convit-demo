"""
Streamlit demo – ConViT inference-only
=====================================

• Downloads the model checkpoint (if missing) from Google Drive
• Loads ConViT architecture from models/hybrid_vit.py
• Accepts a 4-band GeoTIFF upload and predicts multi-label classes
"""

import streamlit as st
import torch, pathlib, subprocess, tempfile, os
import numpy as np, rasterio, cv2

# ────────────────────────────────────
# Basic config
# ────────────────────────────────────
CKPT_NAME: str = "hybrid_ckpt.pt"
CKPT_URL:  str = "https://drive.google.com/file/d/1TXuByjQxxx2gaHMWIHd3LHmjgka8um3M/view?usp=drive_link"
LABELS    = ["cloud", "shadow", "land_cover"]        # ⇐ adapt to your classes
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────
# Download checkpoint once (gdown)
# ────────────────────────────────────
ckpt_path = pathlib.Path(CKPT_NAME)
if not ckpt_path.exists():
    st.info("Downloading checkpoint (~350 MB)…")
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    result = subprocess.run(
        ["gdown", "--fuzzy", "-O", CKPT_NAME, CKPT_URL],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        st.error("Download failed – please check CKPT_URL / File-ID.")
        st.stop()
    st.success("Checkpoint downloaded ✔")

# ────────────────────────────────────
# Model & thresholds (cached)
# ────────────────────────────────────
@st.cache_resource
def load_model():
    # local import – avoids heavy deps if something fails earlier
    from models.hybrid_vit import ConViT

    ckpt = torch.load(CKPT_NAME, map_location=DEVICE)
    model = ConViT(num_classes=len(LABELS))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    # thresholds stored in checkpoint (fallback 0.5)
    thr = ckpt.get("best_thr", np.full(len(LABELS), 0.5))
    return model, thr

model, THRESHOLDS = load_model()

# ────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────
st.set_page_config(page_title="ConViT Demo", layout="centered")
st.title("🛰️ ConViT Satellite Classifier — Demo")

uploaded = st.file_uploader(
    "Upload a **4-band GeoTIFF** chip (R, G, B, NIR)",
    type=("tif", "tiff")
)

if uploaded:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            arr = src.read().astype(np.float32)   # (4, H, W)
    except Exception as e:
        st.error(f"Could not read TIFF: {e}")
        os.remove(tmp_path)
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # resize → 224 × 224 and convert to CHW tensor
    img224 = cv2.resize(arr.transpose(1, 2, 0), (224, 224), cv2.INTER_AREA)
    tensor = torch.from_numpy(img224.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).cpu().numpy()[0]
        pred = (prob >= THRESHOLDS).astype(int)

    st.subheader("Predictions")
    for lab, p, pr in zip(LABELS, prob, pred):
        st.write(f"- **{lab}**: {p:.3f} → {'✅' if pr else '❌'}")

    # display false-colour thumbnail
    nir, green, red = arr[3], arr[1], arr[0]
    thumb = np.stack([nir, green, red], axis=-1)
    thumb = cv2.normalize(thumb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    st.image(thumb, caption="False-colour (NIR-G-R)", channels="RGB")
