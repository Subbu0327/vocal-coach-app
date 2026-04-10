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

for folder in [LIBRARY_DIR, TANPURA_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

st.set_page_config(page_title="VocalVeda AI Studio", layout="wide", page_icon="🎤")

# --- 2. ENHANCED HIGH-CONTRAST & FLOATING UI CSS ---
st.markdown("""
    <style>
    header {visibility: hidden;} footer {visibility: hidden;} [data-testid="stHeader"] {display: none;}
    
    .stApp { 
        background-color: #0f172a;
        background: linear-gradient(-45deg, #0f0c29, #1a1a2e, #16213e); 
        background-size: 400% 400%; 
        animation: gradientBG 15s ease infinite; 
    }

    p, label, .stMarkdown, .stMetric label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    input, div[data-baseweb="select"] > div, .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #00d2ff !important;
        font-weight: 800 !important;
    }

    div[data-baseweb="popover"] ul { background-color: #ffffff !important; }
    div[data-baseweb="popover"] li { color: #000000 !important; font-weight: 600 !important; }
    
    .interactive-card {
        background: rgba(255, 255, 255, 0.08);
        border: 2px solid #00d2ff;
        padding: 25px;
        border-radius: 15px;
    }

    h1, h2, h3 { color: #00d2ff !important; text-shadow: 0 0 10px rgba(0, 210, 255, 0.5); }

    [data-testid="stSidebar"] { 
        background-color: #020617 !important; 
        border-right: 3px solid #00d2ff; 
    }
    
    .stButton>button { 
        background: #0ea5e9 !important; 
        color: white !important; 
        border: 2px solid #ffffff !important;
        font-weight: 800 !important;
    }

    /* --- FLOATING ATTRATIVE CONTACT CARD --- */
    .floating-contact {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 320px;
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid #00d2ff;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 210, 255, 0.4);
        z-index: 1000;
        backdrop-filter: blur(10px);
        animation: slideIn 1s ease-out;
        transition: transform 0.3s ease;
    }
    .floating-contact:hover {
        transform: scale(1.03);
        border-color: #ffd700;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .contact-link {
        color: #00d2ff !important;
        text-decoration: none;
        font-weight: bold;
        display: block;
        margin-top: 10px;
    }
    .contact-link:hover { color: #ffd700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def identify_raga_logic(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096)
    mean_chroma = np.mean(chroma, axis=1)
    tonic_shift = np.argmax(mean_chroma)
    mean_chroma = np.roll(mean_chroma, -tonic_shift)
    if np.max(mean_chroma) > 0:
        mean_chroma = mean_chroma / np.max(mean_chroma)
        mean_chroma[mean_chroma < 0.45] = 0 
    best_match, max_score, coach_tips = "Free-style", -999, "Hold your 'Sa' clearly!"
    for raga, data in RAGA_MAP.items():
        comp_vec = np.array([1 if i in data["notes"] else -2.5 for i in range(12)])
        score = np.dot(mean_chroma, comp_vec)
        if score > max_score: max_score, best_match, coach_tips = score, raga, data["tips"]
    return best_match, coach_tips

def play_bg_audio(file_path, loop=True):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f'<audio autoplay {"loop" if loop else ""}><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# --- 4. INTERFACE ---
tabs = st.tabs(["🎤 Studio Booth", "🎮 Swar-Match", "📈 Leaderboard"])

with tabs[0]:
    st.markdown("<h1 style='text-align: center; color: #00d2ff !important; font-size: 2.5rem;'>VOCALVEDA AI STUDIO</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<h2 style='color: #00d2ff !important;'>🔱 Tanpura Droid</h2>", unsafe_allow_html=True)
        t_files = sorted([f for f in os.listdir(TANPURA_DIR) if f.endswith(('.mp3', '.wav'))])
        if t_files:
            file_map = {f.replace('_', ' ').split('.')[0]: f for f in t_files}
            sel_display = st.selectbox("Select Scale", options=list(file_map.keys()), key="sidebar_scale_select")
            if st.toggle("Power On"): play_bg_audio(os.path.join(TANPURA_DIR, file_map[sel_display]), loop=True)
        st.divider()
        st.markdown("### 💡 Quick Tip")
        st.write("For best results, hold your 'Sa' note for 3 seconds when the recording starts.")

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.markdown("### 1. Setup Session")
        u_name = st.text_input("Singer Name", placeholder="Artist Name...", key="singer_main_input")
        raw_songs = ["Free-style (No Reference)"] + [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
        sel_s = st.selectbox("Select Track / Mode", options=raw_songs, key="track_select_input")
        
        if sel_s == "Free-style (No Reference)":
            st.info("🧘 **Free-style Mode:** Sing anything, the AI will detect your Raga.")
        else:
            st.success(f"🎯 **Challenge:** Matching '{sel_s}'.")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("▶️ HEAR"): st.session_state.playing_ref = True
            with b2:
                if st.button("⏹️ STOP"): st.session_state.playing_ref = False; st.rerun()
            if st.session_state.playing_ref: play_bg_audio(os.path.join(LIBRARY_DIR, sel_s), loop=False)

    with col2:
        st.markdown("### 2. Studio Booth")
        if not u_name.strip():
            st.warning("⚠️ Enter your name in Step 1 to unlock the booth.")
        else:
            st.markdown("<div class='interactive-card'>", unsafe_allow_html=True)
            rec = mic_recorder(start_prompt="Record Vocal 🎙️", stop_prompt="Stop & Analyze ⏹️", key='booth_rec')
            st.markdown("</div>", unsafe_allow_html=True)
            if rec:
                audio_data = AudioSegment.from_file(io.BytesIO(rec['bytes']))
                wav_io = io.BytesIO(); audio_data.export(wav_io, format="wav"); wav_io.seek(0)
                y_u, sr = librosa.load(wav_io, sr=22050)
                st.audio(rec['bytes'])
                if st.button("Generate Performance Report 🚀"):
                    with st.spinner("AI Analysis..."):
                        if sel_s == "Free-style (No Reference)":
                            found_raga, coach_tips = identify_raga_logic(y_u, sr)
                            st.balloons()
                            st.markdown(f"<div class='interactive-card' style='border-color:#ffd700;'><h2 style='color:#ffd700 !important; text-align:center;'>✨ Raga: {found_raga}</h2></div><div style='background:#ffffff; color:#000000; padding:15px; margin-top:10px; border-radius:10px; border:2px solid #1DB954;'><b>💡 COACH TIP:</b> {coach_tips}</div>", unsafe_allow_html=True)
                        else:
                            y_r, _ = librosa.load(os.path.join(LIBRARY_DIR, sel_s), sr=22050)
                            f0_u = librosa.yin(y_u, fmin=80, fmax=800); f0_r = librosa.yin(y_r, fmin=80, fmax=800)
                            L = min(len(f0_u), len(f0_r))
                            m_u, m_r = librosa.hz_to_midi(f0_u[:L]), librosa.hz_to_midi(f0_r[:L])
                            score = round(max(0, 100 - (np.mean(np.where(np.abs((m_u%12)-(m_r%12))<=0.3, 0, np.abs((m_u%12)-(m_r%12))))*15)), 1)
                            st.metric("Vocal Accuracy Score", f"{score}%")
                            if score >= 80: st.balloons()
                            fig, ax = plt.subplots(figsize=(10, 2), facecolor='none'); ax.plot(m_r, color='white', alpha=0.3); ax.plot(m_u, color='#00d2ff', linewidth=2); ax.set_axis_off(); st.pyplot(fig)
                            pd.concat([pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame(), pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d"), "Name": u_name, "Song": sel_s, "Score": score}])], ignore_index=True).to_csv(RESULTS_FILE, index=False)

with tabs[1]:
    st.markdown("<h2 style='text-align: center; color: #1DB954 !important;'>🎮 Swar-Match Challenge</h2>", unsafe_allow_html=True)
    if st.session_state.game_finished:
        st.markdown("<div class='interactive-card' style='border-color:#ffd700; text-align:center;'><h2>🏆 Vocal Champion!</h2><p>You mastered all levels.</p></div>", unsafe_allow_html=True)
        if st.button("Restart Challenge 🔄"):
            st.session_state.game_level = 1; st.session_state.game_finished = False; st.rerun()
    else:
        lvl_data = LEVELS[st.session_state.game_level]
        st.markdown(f"<div class='interactive-card' style='text-align:center;'><h3>Level {st.session_state.game_level}: {lvl_data['name']}</h3><p>{lvl_data['tip']}</p></div>", unsafe_allow_html=True)
        game_rec = mic_recorder(start_prompt="Sing Pattern 🎤", stop_prompt="Submit ⏹️", key='game_rec')
        if game_rec:
            audio_game = AudioSegment.from_file(io.BytesIO(game_rec['bytes']))
            game_wav_io = io.BytesIO(); audio_game.export(game_wav_io, format="wav"); game_wav_io.seek(0)
            y_g, sr = librosa.load(game_wav_io, sr=22050)
            f0 = librosa.yin(y_g, fmin=80, fmax=500); valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                user_midi = np.median(librosa.hz_to_midi(valid_f0))
                dist = abs((user_midi % 12) - (lvl_data["notes"][0] % 12))
                if dist > 6: dist = 12 - dist 
                accuracy = max(0, 100 - (dist * 12)) 
                st.metric("Pitch Accuracy", f"{round(accuracy, 1)}%")
                if accuracy >= 70.0:
                    st.balloons(); st.success("Mastered!"); time.sleep(3)
                    if st.session_state.game_level < 4: st.session_state.game_level += 1
                    else: st.session_state.game_finished = True
                    st.rerun()
                else: st.error("Try again!")

with tabs[2]:
    st.markdown("## 📈 Global Leaderboard")
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE).sort_values(by="Score", ascending=False)
        st.dataframe(df, use_container_width=True)

# --- FLOATING ATTRATIVE CONTACT CARD ---
st.markdown("""
    <div class="floating-contact">
        <h3 style="margin:0; color:#00d2ff;">🔱 Connect with Me</h3>
        <p style="font-size: 0.9rem; margin-top:10px; color:#ffffff;">Drop a message for one-on-one classes and live music sessions!</p>
        <a class="contact-link" href="https://www.instagram.com/subramanya__kashyap?igsh=bTFvbHJ1NTRrMmll&utm_source=qr" target="_blank">📸 INSTAGRAM</a>
        <a class="contact-link" href="mailto:subramanyasambhu@gmail.com">📩 EMAIL ME</a>
    </div>
""", unsafe_allow_html=True)
