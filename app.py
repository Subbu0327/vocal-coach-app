import streamlit as st
import librosa
import numpy as np
import pandas as pd
import os
import io
import base64
import time
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from datetime import datetime
import matplotlib.pyplot as plt

# --- 1. SETUP & CONFIG ---
RESULTS_FILE = "results.csv"
LIBRARY_DIR = "library"
TANPURA_DIR = "tanpura"

for folder in [LIBRARY_DIR, TANPURA_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

LEVELS = {
    1: {"name": "Steady Sa", "notes": [0], "tip": "Hold your base note (Sa) perfectly steady."},
    2: {"name": "Power Pa", "notes": [7], "tip": "Reach the perfect 5th (Pa) relative to your Sa."},
    3: {"name": "Sa-Ga Glide", "notes": [0, 4], "tip": "Move from Sa to the Major 3rd (Ga)."},
    4: {"name": "Vedic Trio", "notes": [0, 2, 4], "tip": "Match the sequence: Sa -> Ri -> Ga."}
}

RAGA_MAP = {
    "Bhairav / Mayamalavagowla": {"notes": {0, 1, 4, 5, 7, 8, 11}, "tips": "Focus on 'Komal Re' and 'Komal Dha'."},
    "Bhoop / Mohanam": {"notes": {0, 2, 4, 7, 9}, "tips": "Pentatonic scale. Avoid 'Ma' and 'Ni'."},
    "Bilawal / Sankarabharanam": {"notes": {0, 2, 4, 5, 7, 9, 11}, "tips": "All Shuddha Swaras. Keep 'Ni' sharp."},
    "Kafi / Kharaharapriya": {"notes": {0, 2, 3, 5, 7, 9, 10}, "tips": "Komal Ga and Komal Ni define this mood."},
    "Hamsadhwani": {"notes": {0, 2, 4, 7, 11}, "tips": "Bright morning Raga. Avoid 'Ma' and 'Dha'."},
    "Malkauns / Hindolam": {"notes": {0, 3, 5, 8, 10}, "tips": "Midnight Raga. Avoid 'Re' and 'Pa'."},
    "Bhairavi (Carnatic)": {"notes": {0, 1, 3, 5, 7, 8, 10}, "tips": "Queen of Ragas. All notes except Sa/Pa are flat."}
}

if 'game_level' not in st.session_state: st.session_state.game_level = 1
if 'game_finished' not in st.session_state: st.session_state.game_finished = False
if 'playing_ref' not in st.session_state: st.session_state.playing_ref = False

st.set_page_config(page_title="VocalVeda AI Studio", layout="wide", page_icon="🎤")

# --- 2. ENHANCED CSS ---
st.markdown("""
    <style>
    header {visibility: hidden;} footer {visibility: hidden;} [data-testid="stHeader"] {display: none;}
    .stApp { background-color: #0f172a; background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%); }
    p, label, .stMarkdown, .stMetric label, h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700 !important; }
    div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
    input, .stTextInput > div > div > input, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input {
        background-color: #ffffff !important; color: #000000 !important; -webkit-text-fill-color: #000000 !important;
        border: 3px solid #00d2ff !important; font-weight: 900 !important; opacity: 1 !important;
    }
    div[data-baseweb="popover"] ul { background-color: #ffffff !important; }
    div[data-baseweb="popover"] li { color: #000000 !important; font-weight: 700 !important; }
    
    /* LEADERBOARD CENTER ALIGNMENT */
    [data-testid="stTable"] th, [data-testid="stTable"] td { text-align: center !important; color: white !important; font-weight: 600 !important; }
    [data-testid="stTable"] th { color: #00d2ff !important; background-color: #020617 !important; }

    .studio-card { background: rgba(255, 255, 255, 0.05); border: 2px solid #00d2ff; padding: 25px; border-radius: 15px; margin-top: 10px; }
    [data-testid="stSidebar"] { background-color: #020617 !important; border-right: 3px solid #00d2ff; }
    .stButton>button { background: #0ea5e9 !important; color: white !important; border: 2px solid #ffffff !important; font-weight: 800 !important; height: 3em !important; border-radius: 10px !important; }
    .contact-container { position: fixed; bottom: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; align-items: flex-end; }
    .contact-menu { display: none; background: rgba(15, 23, 42, 0.98); border: 2px solid #00d2ff; border-radius: 15px; padding: 18px; width: 260px; margin-bottom: 12px; backdrop-filter: blur(15px); box-shadow: 0 10px 40px rgba(0, 210, 255, 0.5); }
    .contact-toggle { background: #00d2ff; color: #000; width: 55px; height: 55px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 26px; cursor: pointer; box-shadow: 0 0 20px #00d2ff; transition: all 0.3s ease; }
    .contact-container:hover .contact-menu { display: block; animation: fadeIn 0.4s ease-out; }
    .contact-container:hover .contact-toggle { transform: rotate(45deg); background: #ffd700; box-shadow: 0 0 20px #ffd700; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    .contact-link { color: #00d2ff !important; text-decoration: none; font-weight: 800; display: block; margin-top: 12px; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC ---
def identify_raga_logic(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096)
    mean_chroma = np.mean(chroma, axis=1)
    tonic_shift = np.argmax(mean_chroma)
    mean_chroma = np.roll(mean_chroma, -tonic_shift)
    best_match, max_score, tips = "Free-style", -999, "Hold your 'Sa' clearly!"
    for raga, data in RAGA_MAP.items():
        comp_vec = np.array([1 if i in data["notes"] else -2.5 for i in range(12)])
        score = np.dot(mean_chroma, comp_vec)
        if score > max_score: max_score, best_match, tips = score, raga, data["tips"]
    return best_match, tips

def play_bg_audio(file_path, loop=True):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f'<audio autoplay {"loop" if loop else ""}><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

def save_score(name, song, score):
    entry = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Name": name, "Song": song, "Score": score}])
    df = pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame()
    pd.concat([df, entry], ignore_index=True).to_csv(RESULTS_FILE, index=False)

# --- 4. INTERFACE ---
tabs = st.tabs(["🎤 Studio Booth", "🎮 Swara-Match", "📈 Leaderboard"])

with tabs[0]:
    st.markdown("<h1 style='text-align: center;'>VOCALVEDA AI STUDIO</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<h2 style='color: #00d2ff !important;'>🪕 Tanpura Droid</h2>", unsafe_allow_html=True)
        t_files = sorted([f for f in os.listdir(TANPURA_DIR) if f.endswith(('.mp3', '.wav'))])
        if t_files:
            file_map = {f.replace('_', ' ').split('.')[0]: f for f in t_files}
            sel_display = st.selectbox("Select Scale", options=list(file_map.keys()), key="sb_scale")
            if st.toggle("Power On"): play_bg_audio(os.path.join(TANPURA_DIR, file_map[sel_display]), loop=True)
        st.divider()
        st.write("### 💡 Quick Tip")
        st.caption("Hold your 'Sa' for 3 seconds to calibrate the AI.")

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.write("### 1. Setup Session")
        u_name = st.text_input("Singer Name", placeholder="Artist Name...", key="artist_input")
        raw_songs = ["Free-style (No Reference)"] + [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
        sel_s = st.selectbox("Mode / Track", options=raw_songs, key="track_select")
        if sel_s != "Free-style (No Reference)":
            c1, c2 = st.columns(2)
            with c1: 
                if st.button("▶️ HEAR REF"): st.session_state.playing_ref = True
            with c2: 
                if st.button("⏹️ STOP REF"): st.session_state.playing_ref = False; st.rerun()
            if st.session_state.playing_ref: play_bg_audio(os.path.join(LIBRARY_DIR, sel_s), loop=False)

    with col2:
        st.write("### 2. Recording Area")
        if not u_name.strip():
            st.warning("👤 Enter Singer Name to unlock.")
        else:
            st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
            rec = mic_recorder(start_prompt="Record Vocal 🎙️", stop_prompt="Analyze Recording ⏹️", key='booth_rec')
            st.markdown("</div>", unsafe_allow_html=True)
            if rec:
                audio_data = AudioSegment.from_file(io.BytesIO(rec['bytes']))
                wav_io = io.BytesIO(); audio_data.export(wav_io, format="wav"); wav_io.seek(0)
                y_u, sr = librosa.load(wav_io, sr=22050)
                st.audio(rec['bytes'])
                if st.button("Generate AI Report 🚀"):
                    if sel_s == "Free-style (No Reference)":
                        found_raga, tips = identify_raga_logic(y_u, sr)
                        st.balloons()
                        st.markdown(f"<div class='studio-card' style='border-color:#ffd700; text-align:center;'><h2>✨ Raga: {found_raga}</h2><p>{tips}</p></div>", unsafe_allow_html=True)
                        save_score(u_name, sel_s, 0)
                    else:
                        y_r, _ = librosa.load(os.path.join(LIBRARY_DIR, sel_s), sr=22050)
                        f0_u, f0_r = librosa.yin(y_u, fmin=80, fmax=800), librosa.yin(y_r, fmin=80, fmax=800)
                        L = min(len(f0_u), len(f0_r))
                        m_u, m_r = librosa.hz_to_midi(f0_u[:L]), librosa.hz_to_midi(f0_r[:L])
                        score = round(max(0, 100 - (np.mean(np.where(np.abs((m_u%12)-(m_r%12))<=0.3, 0, np.abs((m_u%12)-(m_r%12))))*15)), 1)
                        st.metric("Accuracy", f"{score}%")
                        if score >= 80: st.balloons()
                        fig, ax = plt.subplots(figsize=(10, 2), facecolor='none'); ax.plot(m_r, color='white', alpha=0.3); ax.plot(m_u, color='#00d2ff', linewidth=2); ax.set_axis_off(); st.pyplot(fig)
                        save_score(u_name, sel_s, score)

with tabs[1]:
    st.write("## 🎮 Swara-Match Challenge")
    if st.session_state.game_finished:
        st.markdown("<div class='studio-card' style='text-align:center; border-color:#ffd700;'><h2>🏆 Champion!</h2><p>You mastered all levels.</p></div>", unsafe_allow_html=True)
        if st.button("Restart"): st.session_state.game_level = 1; st.session_state.game_finished = False; st.rerun()
    else:
        lvl = LEVELS[st.session_state.game_level]
        st.markdown(f"<div class='studio-card'><h3>Level {st.session_state.game_level}: {lvl['name']}</h3><p>{lvl['tip']}</p></div>", unsafe_allow_html=True)
        g_rec = mic_recorder(start_prompt="Sing Pattern 🎤", stop_prompt="Submit ⏹️", key='game_rec')
        if g_rec:
            audio_game = AudioSegment.from_file(io.BytesIO(g_rec['bytes']))
            game_wav_io = io.BytesIO(); audio_game.export(game_wav_io, format="wav"); game_wav_io.seek(0)
            y_g, sr = librosa.load(game_wav_io, sr=22050)
            f0 = librosa.yin(y_g, fmin=80, fmax=500); v_f0 = f0[~np.isnan(f0)]
            if len(v_f0) > 0:
                user_midi = np.median(librosa.hz_to_midi(v_f0))
                dist = abs((user_midi % 12) - (lvl["notes"][0] % 12))
                accuracy = max(0, 100 - ((12-dist if dist>6 else dist) * 12))
                st.metric("Accuracy", f"{round(accuracy, 1)}%")
                if accuracy >= 70.0:
                    st.balloons(); st.success("Mastered!"); time.sleep(3)
                    if st.session_state.game_level < 4: st.session_state.game_level += 1
                    else: st.session_state.game_finished = True
                    st.rerun()
                else: st.error("Try again!")

with tabs[2]:
    st.write("## 📈 Global Leaderboard")
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE).sort_values(by="Score", ascending=False).reset_index(drop=True)
        df.index += 1
        st.table(df[["Name", "Song", "Score", "Timestamp"]])

st.markdown("""
    <div class="contact-container">
        <div class="contact-menu">
            <h4 style="margin:0; color:#00d2ff;">🔱 Connect with Me</h4>
            <p style="font-size: 0.85rem; margin-top:5px; color:#ffffff;">🎹 Drop a message for <b>One-on-one classes</b>! 🎶</p>
            <a class="contact-link" href="https://www.instagram.com/subramanya__kashyap?igsh=bTFvbHJ1NTRrMmll&utm_source=qr" target="_blank">📸 INSTAGRAM</a>
            <a class="contact-link" href="mailto:subramanyasambhu@gmail.com">📩 EMAIL ME</a>
        </div>
        <div class="contact-toggle">🎹ℹ️</div>
    </div>
""", unsafe_allow_html=True)
