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
from scipy.signal import savgol_filter

# --- SETUP ---
RESULTS_FILE = "results.csv"
LIBRARY_DIR = "library"
TANPURA_DIR = "tanpura"

st.set_page_config(page_title="VocalCoach AI Pro+", layout="wide")

# --- HELPER: ENHANCED FEEDBACK ---
def get_vocal_range(midi_notes):
    """Categorizes the singer based on their average pitch."""
    avg_pitch = np.median(midi_notes)
    if avg_pitch < 50: return "Bass (Low)"
    if avg_pitch < 60: return "Baritone/Tenor (Mid)"
    return "Soprano/Alto (High)"

def play_bg_audio(file_path, loop=True):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<audio autoplay {"loop" if loop else ""}><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)

# --- APP LAYOUT ---
tabs = st.tabs(["🎤 Studio", "📈 Progress Proof"])

with tabs[0]:
    st.title("VocalCoach AI Pro+")
    
    with st.sidebar:
        st.header("🔱 Tanpura Droid")
        t_files = [f for f in os.listdir(TANPURA_DIR) if f.endswith(('.mp3', '.wav'))]
        if t_files:
            if st.toggle("Power Tanpura"):
                play_bg_audio(os.path.join(TANPURA_DIR, st.selectbox("Shruti", t_files)))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("1. Selection")
        u_name = st.text_input("Singer Name", "Guest")
        songs = [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
        if songs:
            sel_s = st.selectbox("Choose Song", songs)
            ref_path = os.path.join(LIBRARY_DIR, sel_s)
            if st.toggle("Play Reference"): play_bg_audio(ref_path, False)

    with col2:
        st.header("2. Analysis")
        rec = mic_recorder(start_prompt="Start Session", stop_prompt="End Session", key='recorder')
        
        if rec:
            st.audio(rec['bytes'])
            if st.button("Deep Analysis 🚀"):
                # Process
                audio = AudioSegment.from_file(io.BytesIO(rec['bytes']))
                wav_io = io.BytesIO(); audio.export(wav_io, format="wav"); wav_io.seek(0)
                y_u, sr = librosa.load(wav_io); y_r, _ = librosa.load(ref_path)
                
                # Analysis
                f0_u = librosa.yin(y_u, fmin=80, fmax=800)
                f0_r = librosa.yin(y_r, fmin=80, fmax=800)
                L = min(len(f0_u), len(f0_r))
                m_u, m_r = librosa.hz_to_midi(f0_u[:L]), librosa.hz_to_midi(f0_r[:L])
                
                # NEW: Vibrato & Range
                v_range = get_vocal_range(m_u)
                diffs = np.abs((m_u % 12) - (m_r % 12))
                score = round(max(0, 100 - (np.mean(np.where(diffs <= 0.3, 0, diffs)) * 15)), 1)
                
                # --- UI ENHANCEMENTS ---
                st.metric("Final Score", f"{score}%", delta=f"{score-50}% vs Avg")
                
                c1, c2 = st.columns(2)
                c1.info(f"**Vocal Type:** {v_range}")
                c2.info(f"**Stability:** {'High' if np.std(m_u) < 2 else 'Needs Work'}")

                # NEW: Heatmap Plot
                fig, ax = plt.subplots(figsize=(10, 3))
                # Create a color map: green for good, red for bad
                colors = ['green' if d < 0.5 else 'orange' if d < 1.0 else 'red' for d in diffs]
                ax.scatter(range(L), m_u, c=colors, s=1, label="Your Pitch (Accuracy Heatmap)")
                ax.plot(m_r, color='gray', alpha=0.3, label="Reference")
                ax.set_ylabel("MIDI Note")
                st.pyplot(fig)

                # Log
                res = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Name": u_name, "Song": sel_s, "Score": score, "Range": v_range}
                if os.path.exists(RESULTS_FILE): pd.concat([pd.read_csv(RESULTS_FILE), pd.DataFrame([res])], ignore_index=True).to_csv(RESULTS_FILE, index=False)
                else: pd.DataFrame([res]).to_csv(RESULTS_FILE, index=False)

with tabs[1]:
    st.header("User Leaderboard")
    if os.path.exists(RESULTS_FILE):
        st.dataframe(pd.read_csv(RESULTS_FILE).sort_values(by="Score", ascending=False), use_container_width=True)