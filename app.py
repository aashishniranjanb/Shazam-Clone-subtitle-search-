import os
import gdown
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import assemblyai as aai
from dotenv import load_dotenv

# Load AssemblyAI API Key from Streamlit secrets
aai.settings.api_key = st.secrets["general"]["ASSEMBLYAI_API_KEY"]

# Download CSV from Google Drive using gdown
GDRIVE_CSV_FILE_ID = "1YlaqKn6DXPu2tySMbo3JKAHJFWLKjjGq"
CSV_PATH = "subtitles.csv"

def download_csv_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.write("Downloading CSV from:", url)
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    return output_path

if not os.path.exists(CSV_PATH):
    with st.spinner("Downloading CSV file from Google Drive..."):
        download_csv_from_drive(GDRIVE_CSV_FILE_ID, CSV_PATH)
        st.success("✅ CSV downloaded successfully!")

st.write(f"CSV file exists: {os.path.exists(CSV_PATH)}")
st.write(f"CSV file size: {os.path.getsize(CSV_PATH)} bytes")


# Load Subtitle Data from CSV
@st.cache_data(show_spinner=True)
def load_subtitles():
    """
    Load subtitles from a CSV file.
    Expected CSV columns: 'num', 'name', 'text'
    """
    df = pd.read_csv(CSV_PATH)
    return df

df = load_subtitles()
if df.empty:
    st.error("Subtitles data is empty!")
else:
    st.success(f"Loaded {len(df)} subtitles.")

@st.cache_data(show_spinner=True)
def compute_embeddings(df):
    """
    Compute embeddings for the subtitle text using SentenceTransformers.
    """
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    return embeddings

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = compute_embeddings(df)

# Streamlit UI: Audio Transcription & Subtitle Search
st.set_page_config(page_title="Audio Transcription & Subtitle Search", layout="wide")
st.title("🎵 Audio Transcription & Subtitle Search")
st.markdown("Upload an **audio file**, transcribe it with AssemblyAI, and find the most similar subtitle segments!")

# Audio file uploader
uploaded_audio = st.file_uploader("🎤 Upload an Audio File", type=["mp3", "wav", "m4a"])
if uploaded_audio:
    st.audio(uploaded_audio, format="audio/mp3")
    
    if st.button("🎙 Transcribe Audio"):
        with st.spinner("Transcribing audio..."):
            # Transcribe audio using AssemblyAI
            transcript = aai.Transcriber().transcribe(uploaded_audio)
            transcribed_text = transcript.text
        st.success("✅ Transcription Complete!")
        st.text_area("🎧 Transcribed Text", transcribed_text, height=150)
        
        query_embedding = model.encode([transcribed_text])
        
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:5]
        
        st.markdown("## 🔍 **Top Matching Subtitles**")
        for idx in top_indices:
            row = df.iloc[idx]
            st.markdown(f"""
            **🎬 Movie:** {row['name']}
            - **📜 Subtitle:** {row['text'][:200]}...
            - 🔗 **[View on OpenSubtitles](https://www.opensubtitles.org/en/subtitles/{row['num']})**
            """)
            

# Sidebar & Footer
st.sidebar.header("🔧 Settings")
st.sidebar.markdown("""
- **Database:** CSV file (subtitles.csv) stored on Google Drive
- **Search Mechanism:** Semantic Search (Cosine Similarity)
""")
st.markdown("---")
st.markdown("**Developed by [Aashish Niranjan BarathyKannan](https://www.linkedin.com/in/aashishniranjanb/)** | [GitHub](https://github.com/aashishniranjanb)")

