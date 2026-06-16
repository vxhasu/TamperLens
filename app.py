import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

st.set_page_config(
    page_title="TamperLens — Label Integrity Checker",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e8eaf0; }
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; border-bottom: 1px solid #1e2130; margin-bottom: 2rem; }
.hero h1 { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -1px; margin: 0 0 0.4rem; }
.hero h1 span { color: #00e5a0; }
.hero p { color: #7b8299; font-size: 1rem; margin: 0; }
.mode-box { background: #161b2e; border: 1px solid #1e2848; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.85rem; color: #7b8299; line-height: 1.6; }
.mode-box b { color: #e8eaf0; }
.mode-badge-classic { display: inline-block; background: #1e2848; color: #7b8299; font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; padding: 3px 10px; border-radius: 20px; margin-bottom: 0.6rem; }
.mode-badge-ai { display: inline-block; background: #0a2e1f; color: #00e5a0; font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; padding: 3px 10px; border-radius: 20px; margin-bottom: 0.6rem; }
.upload-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700; letter-spacing: 2px; color: #00e5a0; text-transform: uppercase; margin-bottom: 0.4rem; }
.result-card { background: #161b2e; border: 1px solid #1e2848; border-radius: 12px; padding: 1.6rem 2rem; margin: 1.5rem 0; }
.score-row { display: flex; align-items: center; gap: 1.2rem; margin-bottom: 0.6rem; }
.score-number { font-family: 'Space Mono', monospace; font-size: 3.2rem; font-weight: 700; line-height: 1; }
.score-label { color: #7b8299; font-size: 0.85rem; }
.verdict-safe { background: #0a2e1f; border: 1px solid #00e5a0; border-radius: 8px; padding: 0.9rem 1.2rem; color: #00e5a0; font-weight: 600; font-size: 1rem; margin-top: 0.8rem; }
.verdict-tampered { background: #2e0a0a; border: 1px solid #ff4d4d; border-radius: 8px; padding: 0.9rem 1.2rem; color: #ff6b6b; font-weight: 600; font-size: 1rem; margin-top: 0.8rem; }
.verdict-warning { background: #2e200a; border: 1px solid #ffa94d; border-radius: 8px; padding: 0.9rem 1.2rem; color: #ffa94d; font-weight: 600; font-size: 1rem; margin-top: 0.8rem; }
.ai-explain { background: #0a1a2e; border: 1px solid #1e3a5f; border-radius: 8px; padding: 0.8rem 1.1rem; margin-top: 0.8rem; font-size: 0.82rem; color: #5b8dd9; font-family: 'Space Mono', monospace; }
.diff-count { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #7b8299; margin-top: 0.5rem; }
.img-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; color: #7b8299; text-transform: uppercase; text-align: center; margin-bottom: 0.3rem; }
.section-title { font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700; letter-spacing: 3px; color: #3d4566; text-transform: uppercase; margin: 1.8rem 0 0.8rem; }
.stButton > button { background: #00e5a0 !important; color: #0f1117 !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 1px !important; border: none !important; border-radius: 8px !important; padding: 0.7rem 2rem !important; width: 100% !important; }
[data-testid="stFileUploader"] { background: #161b2e; border: 1.5px dashed #1e2848; border-radius: 10px; padding: 0.5rem; }
.threshold-note { font-size: 0.78rem; color: #3d4566; margin-top: 0.3rem; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Tamper<span>Lens</span></h1>
    <p>Upload two label images — the system finds differences the human eye can't catch</p>
</div>
""", unsafe_allow_html=True)

# ── MODE TOGGLE ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Detection Mode</div>', unsafe_allow_html=True)
mode = st.radio("", ["⚙️  Classic Mode (SSIM)", "🤖  AI Mode (CLIP)"],
                horizontal=True, label_visibility="collapsed")
use_ai = "AI" in mode

if use_ai:
    st.markdown("""
    <div class="mode-box">
        <div class="mode-badge-ai">AI MODE — CLIP by OpenAI</div><br>
        <b>How it works:</b> CLIP is an AI model made by OpenAI (same company as ChatGPT).
        It was trained on 400 million image-text pairs from the internet.
        It converts each image into 512 numbers that describe its <b>visual meaning</b>.
        Then we compare those numbers — not pixels.<br><br>
        <b>Advantage over Classic:</b> Understands content. A reprinted label that looks
        identical to human eyes but has tiny ink/texture differences? CLIP catches it.
        Also handles slight lighting changes without false alarms.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="mode-box">
        <div class="mode-badge-classic">CLASSIC MODE — SSIM</div><br>
        <b>How it works:</b> Mathematical pixel comparison using Structural Similarity Index.
        Compares luminance, contrast and structure region by region.<br><br>
        <b>Best for:</b> Identical images with small localized edits. Very fast. Shows exact
        pixel-level difference regions with red boxes and heatmap.
    </div>
    """, unsafe_allow_html=True)

# ── AI MODEL (CLIP via sentence-transformers, no torch version issues) ────────
@st.cache_resource
def load_clip_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("clip-ViT-B-32")
    return model

def get_clip_embedding(model, pil_img):
    embedding = model.encode(pil_img, convert_to_numpy=True)
    return embedding

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ── CLASSIC HELPERS ──────────────────────────────────────────────────────────
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return img2
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
    good = matches[:min(200, len(matches))]
    if len(good) < 4:
        return img2
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return img2
    return cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

def compare_classic(img1, img2):
    TARGET = (640, 640)
    img1 = cv2.resize(img1, TARGET)
    img2 = cv2.resize(img2, TARGET)
    img2_aligned = align_images(img1, img2)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2_aligned, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(g1, g2, full=True)
    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 30]
    marked1, marked2 = img1.copy(), img2_aligned.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        pad = 4
        x, y = max(0, x-pad), max(0, y-pad)
        w, h = min(640-x, w+pad*2), min(640-y, h+pad*2)
        cv2.rectangle(marked1, (x,y),(x+w,y+h),(255,60,60),2)
        cv2.rectangle(marked2, (x,y),(x+w,y+h),(255,60,60),2)
    diff_color = cv2.applyColorMap(cv2.bitwise_not(diff), cv2.COLORMAP_INFERNO)
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    return score * 100, contours, marked1, marked2, diff_color

def arr_to_pil(arr):
    return Image.fromarray(arr.astype("uint8"))

# ── UPLOAD ───────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.markdown('<div class="upload-label">Original / Reference Image</div>', unsafe_allow_html=True)
    file1 = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"], key="img1", label_visibility="collapsed")
with col_r:
    st.markdown('<div class="upload-label">Suspected / Comparison Image</div>', unsafe_allow_html=True)
    file2 = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"], key="img2", label_visibility="collapsed")

st.markdown('<div class="section-title">Detection Sensitivity</div>', unsafe_allow_html=True)
threshold = st.slider("", min_value=80, max_value=99, value=95, label_visibility="collapsed")
st.markdown(f'<div class="threshold-note">FLAG as tampered if similarity &lt; {threshold}% &nbsp;|&nbsp; WARN between {threshold}%–{threshold+2}%</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
run = st.button("🔍  Analyse Images")

if run:
    if not file1 or not file2:
        st.error("Please upload both images before analysing.")
    else:
        img1_np = load_image(file1)
        img2_np = load_image(file2)
        pil1 = Image.fromarray(img1_np)
        pil2 = Image.fromarray(img2_np)

        # Always run classic for the visual diff map
        with st.spinner("Analysing pixel differences…"):
            classic_score, contours, marked1, marked2, diff_map = compare_classic(img1_np, img2_np)

        if use_ai:
            with st.spinner("🤖 Loading CLIP AI model (first time ~30 seconds to download)…"):
                clip_model = load_clip_model()
            with st.spinner("🤖 Extracting AI features and comparing…"):
                emb1 = get_clip_embedding(clip_model, pil1)
                emb2 = get_clip_embedding(clip_model, pil2)
                raw_cos = cosine_similarity(emb1, emb2)
                # CLIP cosine scores are typically 0.85-1.0 for similar images
                # Rescale to 0-100 range that makes intuitive sense
                similarity = max(0.0, (raw_cos - 0.5) / 0.5) * 100
        else:
            similarity = classic_score

        num_regions = len(contours)
        score_color = "#00e5a0" if similarity >= threshold+2 else ("#ffa94d" if similarity >= threshold else "#ff4d4d")

        if similarity >= threshold+2:
            verdict_class, verdict_icon = "verdict-safe", "✅"
            verdict_text = f"AUTHENTIC — No significant tampering detected ({similarity:.2f}% match)"
            ai_note = "CLIP found these images semantically identical. Same label content confirmed."
        elif similarity >= threshold:
            verdict_class, verdict_icon = "verdict-warning", "⚠️"
            verdict_text = f"SUSPICIOUS — Minor differences detected ({similarity:.2f}% match)"
            ai_note = "CLIP detected slight content differences. Could be lighting or actual tampering."
        else:
            verdict_class, verdict_icon = "verdict-tampered", "🚨"
            verdict_text = f"TAMPERED — Significant differences detected ({similarity:.2f}% match)"
            ai_note = "CLIP sees these as meaningfully different images. Tampering likely."

        if use_ai:
            st.markdown(f"""
            <div class="result-card">
                <div class="score-row">
                    <div class="score-number" style="color:{score_color}">{similarity:.1f}%</div>
                    <div>
                        <div style="color:#e8eaf0;font-weight:600;font-size:1.05rem">AI Similarity Score</div>
                        <div class="score-label">CLIP (OpenAI) · {num_regions} pixel-diff region(s) also found</div>
                    </div>
                </div>
                <div class="{verdict_class}">{verdict_icon} &nbsp; {verdict_text}</div>
                <div class="ai-explain">🧠 AI says: {ai_note}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="score-row">
                    <div class="score-number" style="color:{score_color}">{similarity:.1f}%</div>
                    <div>
                        <div style="color:#e8eaf0;font-weight:600;font-size:1.05rem">Similarity Score</div>
                        <div class="score-label">SSIM (Structural Similarity Index) · {num_regions} difference region(s)</div>
                    </div>
                </div>
                <div class="{verdict_class}">{verdict_icon} &nbsp; {verdict_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="img-label">Original (marked)</div>', unsafe_allow_html=True)
            st.image(arr_to_pil(marked1), use_container_width=True)
        with c2:
            st.markdown('<div class="img-label">Suspected (marked)</div>', unsafe_allow_html=True)
            st.image(arr_to_pil(marked2), use_container_width=True)
        with c3:
            st.markdown('<div class="img-label">Difference Heatmap</div>', unsafe_allow_html=True)
            st.image(arr_to_pil(diff_map), use_container_width=True)

        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        buf = io.BytesIO()
        arr_to_pil(diff_map).save(buf, format="PNG")
        st.download_button("⬇  Download Difference Map", data=buf.getvalue(),
                           file_name="tamperlens_diff.png", mime="image/png")
