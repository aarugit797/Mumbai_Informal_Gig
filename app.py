import streamlit as st
from streamlit_mic_recorder import speech_to_text

# Set page config FIRST
st.set_page_config(page_title="Mumbai Worker Sahayak", page_icon="🎙️")

# UI Layout
st.title("👷 Mumbai Worker Sahayak")
st.write("Welcome! This AI helps you find information regarding labor laws, worker rights, and more in Mumbai.")

# lazy load the engine to ensure Streamlit's UI renders first
@st.cache_resource
def load_engine():
    from engine import rag_chain
    return rag_chain

# --- INPUT SECTION ---
st.write("### 🎤 Ask your question:")
st.write("Speak your question (Hindi/Marathi/English):")
text = speech_to_text(language='hi-IN', start_prompt="⏺️ Start Speaking", stop_prompt="⏹️ Stop", key='STT')

# --- TEXT INPUT ---
typed_query = st.text_input("Or type your question here:", value=text if text else "")

if st.button("Get Help") and typed_query:
    with st.spinner("AI is analyzing your request..."):
        try:
            # Load engine (will use cached version if available)
            chain = load_engine()
            
            # Invoke chain
            response = chain.invoke(typed_query)
            
            st.markdown(f"### 📢 Assistance:")
            st.info(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Tip: If you just deployed, the AI might still be initializing its database. Please wait a moment and try again.")