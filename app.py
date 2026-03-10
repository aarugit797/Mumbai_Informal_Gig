__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from streamlit_mic_recorder import speech_to_text
import import_ipynb
from engine import rag_chain  

st.set_page_config(page_title="Mumbai Worker Sahayak", page_icon="🎙️")
st.title("👷 Mumbai Worker Sahayak")

# --- VOICE INPUT ---
st.write("Speak your question (Hindi/Marathi/English):")
text = speech_to_text(language='hi-IN', start_prompt="⏺️ Start Speaking", stop_prompt="⏹️ Stop", key='STT')

# --- TEXT INPUT ---
typed_query = st.text_input("Or type your question here:", value=text if text else "")

if st.button("Get Help") and typed_query:
    with st.spinner("AI is reading your documents..."):
        try:
            response = rag_chain.invoke(typed_query)
            st.markdown(f"### 📢 Assistance:\n{response}")
        except Exception as e:
            st.error(f"Error: {str(e)}")