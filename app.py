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

# Create folders if they don't exist
for folder in [LIBRARY_DIR, TANPURA_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

st.set_page_config(page_title="VocalCoach AI Studio", layout="wide", page_icon="🎤")

# --- 2. HELPER FUNCTIONS ---
def play_bg_audio(file_path, loop=True):
    """Background audio player using HTML."""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        loop_attr = "loop" if loop else ""
        audio_html = f"""<audio autoplay {loop_attr}><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
        st.markdown(audio_html, unsafe_allow_html=True)

# --- 3. APP LAYOUT ---
tabs = st.tabs(["🎤 Live Studio", "📈 Progress Proof"])

with tabs[0]:
    st.title("VocalCoach AI 🎶")
    
    # --- SIDEBAR: TANPURA ---
    with st.sidebar:
        st.header("🔱 Tanpura Droid")
        t_files = [f for f in os.listdir(TANPURA_DIR) if f.endswith(('.mp3', '.wav'))]
        if t_files:
            sel_t = st.selectbox("Select Shruti", t_files)
            if st.toggle("Power Tanpura"):
                play_bg_audio(os.path.join(TANPURA_DIR, sel_t), loop=True)
    
    col1, col2 = st.columns([1, 2])
    
    # --- SECTION 1: SELECTION ---
    with col1:
        st.header("1. Selection")
        u_name = st.text_input("Singer Name", value="", placeholder="Enter your name here...")
        
        raw_songs = [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
        song_options = ["Choose the song which you want to record"] + raw_songs
        
        sel_s = st.selectbox("Song Library", options=song_options)
        song_selected = sel_s != "Choose the song which you want to record"
        
        if song_selected:
            st.info("🎵 Listen carefully, then sing with your heart!")
            ref_path = os.path.join(LIBRARY_DIR, sel_s)
            if st.toggle("Play Reference"): 
                play_bg_audio(ref_path, loop=False)

    # --- SECTION 2: RECORDING BOOTH ---
    with col2:
        st.header("2. Recording Booth")
        recording_container = st.container(border=True)
        
        with recording_container:
            if not u_name.strip() or not song_selected:
                st.warning("⚠️ Please complete Section 1 to enable the recording booth.")
                rec = None
            else:
                st.markdown(f"**👤 Singer:** {u_name} | **🎵 Song:** {sel_s}")
                st.write("Please click **Start Session** to record your voice.")
                rec = mic_recorder(start_prompt="Start Session 🎙️", stop_prompt="End Session ⏹️", key='recorder')

        if rec:
            # Graphical Waveform Review
            with st.expander("📊 Performance Waveform", expanded=True):
                audio = AudioSegment.from_file(io.BytesIO(rec['bytes']))
                wav_io = io.BytesIO(); audio.export(wav_io, format="wav"); wav_io.seek(0)
                y_u, sr = librosa.load(wav_io, sr=22050)
                
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(y_u, color='#1DB954', alpha=0.7)
                ax.set_axis_off()
                st.pyplot(fig)
                st.audio(rec['bytes'])
            
            # Deep Analysis
            if st.button("Deep Analysis 🚀"):
                with st.spinner("Processing your results..."):
                    y_r, _ = librosa.load(ref_path, sr=22050)
                    f0_u = librosa.yin(y_u, fmin=80, fmax=800)
                    f0_r = librosa.yin(y_r, fmin=80, fmax=800)
                    L = min(len(f0_u), len(f0_r))
                    m_u, m_r = librosa.hz_to_midi(f0_u[:L]), librosa.hz_to_midi(f0_r[:L])
                    
                    diffs = np.abs((m_u % 12) - (m_r % 12))
                    score = round(max(0, 100 - (np.mean(np.where(diffs <= 0.3, 0, diffs)) * 15)), 1)
                    
                st.metric("Final Score", f"{score}%")

                # Results Heatmap
                fig_res, ax_res = plt.subplots(figsize=(10, 4))
                colors = ['#28a745' if d < 0.5 else '#ffc107' if d < 1.0 else '#dc3545' for d in diffs]
                ax_res.scatter(range(L), m_u, c=colors, s=2)
                ax_res.plot(m_r, color='gray', alpha=0.3, linestyle='--')
                ax_res.set_title("Pitch Accuracy Heatmap (Green = Correct)")
                st.pyplot(fig_res)

                # Save Data for Leaderboard
                res = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Name": u_name, "Song": sel_s, "Score": score}
                pd.concat([pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame(), pd.DataFrame([res])], ignore_index=True).to_csv(RESULTS_FILE, index=False)

    # --- 4. BOTTOM RIGHT CONTACT FOOTER ---
    st.markdown("---")
    _, _, footer_col = st.columns([1, 1, 1.2])
    
    with footer_col:
        with st.container(border=True):
            st.markdown("### 📞 Connect with Me")
            
            # Email Clickable Link
            st.markdown(
                f'<a href="mailto:subramanyasambhu@gmail.com" style="text-decoration: none;">'
                f'<div style="border-radius: 5px; padding: 8px; margin-bottom: 5px; background-color: #f0f2f6; color: #31333F; font-size: 14px; border: 1px solid #dcdfe3; text-align: center;">'
                f'📩 subramanyasambhu@gmail.com'
                f'</div></a>', 
                unsafe_allow_html=True
            )
            
            # Instagram Clickable Link
            st.markdown(
                f'<a href="https://www.instagram.com/subramanya__kashyap?igsh=bTFvbHJ1NTRrMmll&utm_source=qr" target="_blank" style="text-decoration: none;">'
                f'<div style="border-radius: 5px; padding: 8px; margin-bottom: 5px; background-color: #f0f2f6; color: #31333F; font-size: 14px; border: 1px solid #dcdfe3; text-align: center;">'
                f'📸 Follow on Instagram, check out my recent content! and go listen for my covers!'
                f'</div></a>', 
                unsafe_allow_html=True
            )
            
            st.caption("🎓 **Personal Training:** Contact me via Instagram for information on singing classes!")

# --- 5. PROGRESS PROOF TAB ---
with tabs[1]:
    st.header("User Leaderboard")
    if os.path.exists(RESULTS_FILE):
        history_df = pd.read_csv(RESULTS_FILE)
        st.dataframe(history_df[["Timestamp", "Name", "Song", "Score"]].sort_values(by="Score", ascending=False), use_container_width=True)
    else:
        st.info("No records found yet. Be the first to sing!")
