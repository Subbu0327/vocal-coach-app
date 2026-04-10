import streamlit as st
import librosa
import numpy as np
import pandas as pd
import os
import io
import base64
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from datetime import datetime
import matplotlib.pyplot as plt

# --- 1. SETUP & CONFIG ---
RESULTS_FILE = "results.csv"
LIBRARY_DIR = "library"
TANPURA_DIR = "tanpura"

for folder in [LIBRARY_DIR, TANPURA_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

if 'playing_ref' not in st.session_state:
    st.session_state.playing_ref = False

st.set_page_config(page_title="VocalCoach AI Pro", layout="wide", page_icon="🎤")

# --- 2. STUDIO UI CSS (WITH BRANDING SHIELD) ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT BRANDING & TOOLBARS */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    .practice-card {
        background: rgba(0, 210, 255, 0.15);
        padding: 25px;
        border-radius: 20px;
        border: 2px solid #00d2ff;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
        margin-bottom: 15px;
    }
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 2px solid #00d2ff;
    }
    .stMarkdown, p, label, .stMetric label {
        color: #ffffff !important;
    }
    .stButton>button {
        background: #1DB954 !important;
        color: white !important;
        border: 2px solid #ffffff !important;
        font-weight: 900 !important;
        width: 100% !important;
        border-radius: 12px !important;
        height: 3.5em !important;
    }
    .st-emotion-cache-p5msec {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def play_bg_audio(file_path, loop=True):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        loop_attr = "loop" if loop else ""
        audio_html = f"""<audio autoplay {loop_attr}><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
        st.markdown(audio_html, unsafe_allow_html=True)

# --- 4. STUDIO INTERFACE ---
tabs = st.tabs(["🎤 Studio Booth", "📈 Leaderboard"])

with tabs[0]:
    st.markdown("<h1 style='color: #00d2ff; text-align: center; text-shadow: 0 0 15px #00d2ff;'>VocalCoach AI Pro</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h2 style='color: #00d2ff;'>🔱 Tanpura Droid</h2>", unsafe_allow_html=True)
        t_files = [f for f in os.listdir(TANPURA_DIR) if f.endswith(('.mp3', '.wav'))]
        if t_files:
            t_files.sort()
            display_names = [f.replace('_', ' ').replace('.mp3', '').replace('.wav', '') for f in t_files]
            file_map = dict(zip(display_names, t_files))
            sel_display = st.selectbox("Select Scale", options=display_names)
            if st.toggle("Power On"):
                play_bg_audio(os.path.join(TANPURA_DIR, file_map[sel_display]), loop=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### 1. Setup Session")
        u_name = st.text_input("Stage Name", value="", placeholder="Artist Name...")
        raw_songs = [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
        sel_s = st.selectbox("Select Track", options=["Choose a song"] + raw_songs)
        
        if sel_s != "Choose a song":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div class="practice-card"><h4 style="color:#00d2ff; margin:0;">🎧 Practice Mode</h4></div>""", unsafe_allow_html=True)
            ref_path = os.path.join(LIBRARY_DIR, sel_s)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("▶️ PLAY"): st.session_state.playing_ref = True
            with b2:
                if st.button("⏹️ STOP"): 
                    st.session_state.playing_ref = False
                    st.rerun()
            if st.session_state.playing_ref: play_bg_audio(ref_path, loop=False)

    with col2:
        st.markdown("### 2. Live Recording")
        if not u_name.strip() or sel_s == "Choose a song":
            st.info("⚡ Complete Setup in Section 1 to unlock the studio booth.")
        else:
            rec = mic_recorder(start_prompt="Record Vocal 🎙️", stop_prompt="Stop Recording ⏹️", key='recorder')

            if rec:
                st.success("✅ Capture Complete")
                with st.expander("📊 Vocal Signature Review", expanded=True):
                    audio = AudioSegment.from_file(io.BytesIO(rec['bytes']))
                    wav_io = io.BytesIO(); audio.export(wav_io, format="wav"); wav_io.seek(0)
                    y_u, sr = librosa.load(wav_io, sr=22050)
                    
                    fig_wave, ax_wave = plt.subplots(figsize=(10, 2), facecolor='none')
                    ax_wave.plot(y_u, color='#00d2ff', linewidth=1)
                    ax_wave.set_axis_off()
                    st.pyplot(fig_wave)
                    st.audio(rec['bytes'])
                
                if st.button("Generate Performance Report 🚀"):
                    rms = librosa.feature.rms(y=y_u)
                    avg_volume = np.mean(rms)

                    if avg_volume < 0.015:
                        st.error("⚠️ Silent Recording Detected!")
                        st.warning("The AI detected mostly silence. Please sing closer to the mic to get an accurate score.")
                    else:
                        with st.spinner("AI Decoding Vocal Frequencies..."):
                            y_r, _ = librosa.load(os.path.join(LIBRARY_DIR, sel_s), sr=22050)
                            f0_u = librosa.yin(y_u, fmin=80, fmax=800)
                            f0_r = librosa.yin(y_r, fmin=80, fmax=800)
                            L = min(len(f0_u), len(f0_r))
                            m_u, m_r = librosa.hz_to_midi(f0_u[:L]), librosa.hz_to_midi(f0_r[:L])
                            diffs = np.abs((m_u % 12) - (m_r % 12))
                            pitch_score = round(max(0, 100 - (np.mean(np.where(diffs <= 0.3, 0, diffs)) * 15)), 1)
                        
                        st.metric("Pitch Accuracy", f"{pitch_score}%")
                        if pitch_score >= 80: st.balloons()
                        
                        fig_res, ax_res = plt.subplots(figsize=(12, 4), facecolor='none')
                        t = np.linspace(0, len(y_u)/sr, L)
                        ax_res.plot(t, m_r, color='gray', alpha=0.3)
                        ax_res.plot(t, m_u, color='#00d2ff', linewidth=2)
                        ax_res.fill_between(t, m_r, m_u, where=(diffs <= 0.5), color='cyan', alpha=0.2)
                        ax_res.set_axis_off()
                        st.pyplot(fig_res)

                        res = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Name": u_name, "Song": sel_s, "Score": pitch_score}
                        pd.concat([pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame(), pd.DataFrame([res])], ignore_index=True).to_csv(RESULTS_FILE, index=False)

    # --- FOOTER ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, _, footer_col = st.columns([1, 1, 1.2])
    with footer_col:
        st.markdown("""
            <div style="background:rgba(0,210,255,0.1); padding:25px; border-radius:20px; border:1px solid #00d2ff; text-align:center;">
                <h3 style="margin:0; color:#00d2ff;">📞 Connect</h3>
                <div style="margin-top: 15px;">
                    <p style="color:white; margin:10px 0; font-size:15px;">
                        🎓 <b>Personal Training:</b> Contact me via Instagram for information on singing classes!
                    </p>
                    <a href="mailto:subramanyasambhu@gmail.com" style="color:white; text-decoration:none; display:block; margin:10px 0; font-size:16px;">
                        📩 <b>EMAIL ME</b>
                    </a>
                    <a href="https://www.instagram.com/subramanya__kashyap?igsh=bTFvbHJ1NTRrMmll&utm_source=qr" target="_blank" style="color:white; text-decoration:none; display:block; margin:10px 0; font-size:16px;">
                        📸 <b>INSTAGRAM</b>
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<h2 style='color: #00d2ff;'>📈 Global Leaderboard</h2>", unsafe_allow_html=True)
    if os.path.exists(RESULTS_FILE):
        history_df = pd.read_csv(RESULTS_FILE)
        st.dataframe(history_df[["Timestamp", "Name", "Song", "Score"]].sort_values(by="Score", ascending=False), use_container_width=True)
