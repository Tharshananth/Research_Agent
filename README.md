# Smart Multi-Agent System for Research Paper

### output  video [https://drive.google.com/file/d/1_hBwtjRfjP-xhPezZ3UWveqjRQWQtIDN/view?usp=sharing](https://drive.google.com/file/d/1_hBwtjRfjP-xhPezZ3UWveqjRQWQtIDN/view?usp=sharing)

## Smart Multi-Agent System for Research Paper Analysis and Podcast Generation
### Why:
To make academic research more accessible by automating paper retrieval, summarization, classification, and converting summaries into podcast-style audio for easy consumption by researchers and students.

### What:
Developed an AI-powered multi-agent system that fetches research papers, analyzes their content, generates summaries using LLMs, synthesizes topic-wise insights, and converts them into audio format.

### How:

Built agents for fetching, extraction, classification, summarization, synthesis, and audio generation.

Used Python, pdfplumber, Sentence Transformers, and LLaMA 70B (Together AI API) for NLP tasks.

Implemented gTTS and Sarvam TTS API for text-to-speech conversion.

Output included structured JSON, topic summaries, and MP3 audio files.

### Results:

Automated the entire research paper workflow, reducing manual effort by 70%.

Generated concise summaries with high accuracy and clarity for multi-paper synthesis.

Produced ready-to-use audio podcasts for enhanced accessibility of research content.
### How to Execute
Clone the repository

```bash
git clone https://github.com/Tharshananth/vahanAI-.git
cd vahanAI-
```



####  backend
```bash
pip install -r requirements.txt
```

####  Install dependencies

```bash
python3 run.oy 
```

####  (Optional) Start the FastAPI backend


```bash
uvicorn backend:app --reload
```

####  (Optional) Start the FastAPI backend


```bash
streamlit run app.py

```


