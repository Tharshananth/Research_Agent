import streamlit as st
from gtts import gTTS
from io import BytesIO
import os
import json

# Page setup
st.set_page_config(page_title="🧠 Research AI Chat", layout="centered")
st.title("💬 AI Research Paper Assistant")

st.markdown("Ask a question or type a topic to summarize research papers:")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
query = st.text_input("🗨 You:", placeholder="e.g., Summarize recent research on quantum computing")

if st.button("Send") and query:
    st.session_state.history.append(("user", query))

    with st.spinner("Thinking..."):
        try:
            # Run the backend script (run.py must write to topic_syntheses.txt and research_output.json)
            os.system("python run.py")

            # Base directory
            base_dir = os.path.dirname(os.path.abspath(__file__))

            # === Read summary from topic_syntheses.txt ===
            summary_path = os.path.join(base_dir, "..", "topic_syntheses.txt")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read().strip()

            # === Convert summary to audio ===
            tts = gTTS(summary)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)

            # === Read research_output.json for links and pdf_url ===
            json_path = os.path.join(base_dir, "..", "research_output.json")
            with open(json_path, "r", encoding="utf-8") as jf:
                research_data = json.load(jf)
                links = research_data.get("links", [])
                pdf_url = research_data.get("pdf_url", "")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

        # Save the response to session state
        st.session_state.history.append(("ai", summary, links, pdf_url, mp3_fp))

# Display chat history
for msg in st.session_state.history:
    if msg[0] == "user":
        st.markdown(f"🧑‍💻 You:** {msg[1]}")
    else:
        summary, links, pdf_url, mp3_fp = msg[1], msg[2], msg[3], msg[4]
        st.markdown(f"🤖 AI:** {summary}")

        # Audio playback + download
        try:
            st.audio(mp3_fp.getvalue(), format="audio/mp3")
            st.download_button(
                label="🔊 Download Summary Audio",
                data=mp3_fp,
                file_name="summary.mp3",
                mime="audio/mp3"
            )
        except Exception as e:
            st.warning(f"Audio playback failed: {e}")

        # References (from JSON)
        if links:
            st.markdown("🔗 *References from Research Output:*")
            for i, link in enumerate(links, 1):
                st.markdown(f"{i}. [Link]({link})")

        # PDF Download (from JSON)
        if pdf_url:
            st.markdown(f"📄 [Download Related Research PDF]({pdf_url})", unsafe_allow_html=True)

st.markdown("---")
