import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os
import textstat
import numpy as np
import requests
import re
import fitz  # PyMuPDF
from docx import Document
from gtts import gTTS
from bs4 import BeautifulSoup

st.set_page_config(page_title="ToneTune Academic", layout="wide")

# --- Custom CSS for Modern Gradient and Cards ---
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%) !important;
    font-family: 'Quicksand', sans-serif;
}
.tone-header {
    text-align: center;
    color: #fff;
    font-weight: bold;
    font-size: 2.5em;
    margin-top: 0.5em;
    margin-bottom: 0.2em;
    letter-spacing: 1px;
    text-shadow: 0 2px 8px #185a9d99;
}
.tone-sub {
    text-align: center;
    color: #e0e0e0;
    font-size: 1.2em;
    margin-bottom: 2em;
}
.tone-card {
    background: rgba(255,255,255,0.10);
    border-radius: 18px;
    box-shadow: 0 4px 24px #185a9d33;
    padding: 2.5em 2em 2em 2em;
    margin-bottom: 2em;
    color: #fff;
}
.stTextArea textarea {
    background: #f7fafc !important;
    border-radius: 10px !important;
    border: 1.5px solid #43cea2 !important;
    color: #222 !important;
    font-size: 1.1em !important;
}
.stButton>button {
    background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%) !important;
    color: #fff !important;
    border-radius: 12px !important;
    border: none !important;
    font-size: 1.08em !important;
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 4px #185a9d44;
    transition: 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #185a9d 0%, #43cea2 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px #43cea299;
}
.stDownloadButton>button {
    background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%) !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align:center; color:#43cea2; font-size:2em; font-weight:bold; margin-bottom:0.5em;'>ToneTune</div>
<div style='text-align:center; color:#888; font-size:1.1em; margin-bottom:1em;'>Academic Writing Assistant</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📄 Upload a document", type=["txt", "docx", "pdf"])

feature = st.sidebar.radio(
    "Choose a feature:",
    ["Grammar & Style", "Sentiment & Tone", "Readability", "Plagiarism", "Citation Assistance"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Your academic writing assistant for clarity, correctness, and confidence."
)

if st.sidebar.button("Reset"):
    st.rerun()

# --- HEADER ---
st.markdown("<div class='tone-header'>ToneTune Academic</div>", unsafe_allow_html=True)
st.markdown("<div class='tone-sub'>A modern, focused writing assistant for clarity, correctness, and confidence.</div>", unsafe_allow_html=True)

# Move text area here, under the header and subtitle
def get_text():
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            try:
                text = uploaded_file.read().decode("utf-8")
            except Exception:
                text = "(Unable to decode file. Please upload a .txt file, .pdf, or paste text below.)"
    else:
        text = st.text_area("Paste your text here", height=200)
    return text

text = get_text()

# --- MAIN FEATURE PANELS ---
def card(content):
    st.markdown(f"<div class='tone-card'>{content}</div>", unsafe_allow_html=True)

def basic_sent_tokenize(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# --- GRAMMAR & STYLE ---
if feature == "Grammar & Style":
    card("""
    <h3>Grammar & Style Correction</h3>
    <p>Check and improve your grammar and style. Paste or upload your text, then click Analyze.</p>
    """)
    @st.cache_resource(show_spinner=True)
    def load_grammar_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction").to(device)
        tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
        return model, tokenizer, device
    model, tokenizer, device = load_grammar_model()
    def fix_grammar(text):
        sentences = basic_sent_tokenize(text)
        corrected_sentences = []
        for sentence in sentences:
            input_text = "fix grammar: " + sentence.strip()
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True
            ).to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_sentences.append(corrected)
        return " ".join(corrected_sentences)
    if st.button("Analyze Grammar & Style"):
        if not text.strip():
            st.warning("Please provide some text for analysis.")
        else:
            with st.spinner("Analyzing grammar..."):
                corrected = fix_grammar(text)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text**")
                st.write(text)
            with col2:
                st.markdown("**Corrected Text**")
                st.write(corrected)
            st.success("Grammar correction complete!")

# --- SENTIMENT & TONE ---
if feature == "Sentiment & Tone":
    card("""
    <h3>Sentiment & Tone Analysis</h3>
    <p>Analyze the sentiment and tone of your text.</p>
    """)
    @st.cache_resource(show_spinner=True)
    def load_sentiment_model():
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        labels = ["Negative", "Neutral", "Positive"]
        return model, tokenizer, labels
    sent_model, sent_tokenizer, sent_labels = load_sentiment_model()
    def get_sentiment(text):
        import torch.nn.functional as F
        sentences = basic_sent_tokenize(text)
        sentiments = []
        confidences = []
        for sent in sentences:
            inputs = sent_tokenizer(sent, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = sent_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                sentiment = torch.argmax(probs, dim=1).item()
                sentiments.append(sent_labels[sentiment])
                confidences.append(float(probs[0][sentiment].item()))
        if sentiments:
            from collections import Counter
            majority = Counter(sentiments).most_common(1)[0][0]
            avg_conf = np.mean(confidences)
            return majority, avg_conf
        else:
            return "N/A", 0.0
    if st.button("Analyze Sentiment & Tone"):
        if not text.strip():
            st.warning("Please provide some text for analysis.")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = get_sentiment(text)
            st.markdown(f"**Majority Sentiment:** {sentiment}")
            st.markdown(f"**Average Confidence:** {confidence:.2f}")

# --- READABILITY ---
if feature == "Readability":
    card("""
    <h3>Readability Scoring & Simplification</h3>
    <p>Get readability scores and grade level predictions for your text.</p>
    """)
    @st.cache_resource(show_spinner=True)
    def load_readability_model():
        model_name = "agentlans/deberta-v3-xsmall-readability"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    read_model, read_tokenizer, read_device = load_readability_model()
    def predict_readability(text):
        inputs = read_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(read_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = read_model(**inputs).logits.squeeze().cpu()
        grade = float(logits.item())
        return round(grade, 2)
    if st.button("Analyze Readability"):
        if not text.strip():
            st.warning("Please provide some text for analysis.")
        else:
            with st.spinner("Analyzing readability..."):
                grade = predict_readability(text)
                flesch = textstat.flesch_reading_ease(text)
                fog = textstat.gunning_fog(text)
            st.markdown(f"**Predicted Grade Level:** {grade}")
            st.markdown(f"**Flesch Reading Ease:** {flesch:.2f}")
            st.markdown(f"**Gunning Fog Index:** {fog:.2f}")

# --- PLAGIARISM ---
if feature == "Plagiarism":
    card("""
    <h3>Plagiarism Detection</h3>
    <p>Check your text against reference sources for possible plagiarism.</p>
    """)
    ref_texts = st.text_area("Reference Texts (one per line)", height=100)
    ref_file = st.file_uploader("Upload reference file(s)", type=["txt"], key="plag_ref")
    reference_texts = []
    if ref_texts:
        reference_texts.extend([t for t in ref_texts.split("\n") if t.strip()])
    if ref_file:
        try:
            file_text = ref_file.read().decode("utf-8")
            reference_texts.append(file_text)
        except Exception:
            st.warning("Could not read uploaded reference file.")
    @st.cache_resource(show_spinner=True)
    def load_plagiarism_models():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model_large = SentenceTransformer('all-mpnet-base-v2')
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert_model = AutoModel.from_pretrained('bert-base-uncased')
        return model, model_large, bert_tokenizer, bert_model
    s_model, s_model_large, bert_tokenizer, bert_model = load_plagiarism_models()
    def calculate_semantic_similarity(text1, text2, use_large_model=False):
        model = s_model_large if use_large_model else s_model
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    def calculate_bert_similarity(text1, text2):
        inputs1 = bert_tokenizer(text1, return_tensors='pt', max_length=512, truncation=True, padding=True)
        inputs2 = bert_tokenizer(text2, return_tensors='pt', max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs1 = bert_model(**inputs1)
            outputs2 = bert_model(**inputs2)
            emb1 = outputs1.last_hidden_state.mean(dim=1)
            emb2 = outputs2.last_hidden_state.mean(dim=1)
        similarity = cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]
        return float(similarity)
    def calculate_lexical_similarity(text1, text2):
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    def plagiarism_check(suspected_text, reference_texts, threshold=0.75):
        results = []
        for ref in reference_texts:
            sem_sim = calculate_semantic_similarity(suspected_text, ref)
            sem_sim_large = calculate_semantic_similarity(suspected_text, ref, use_large_model=True)
            bert_sim = calculate_bert_similarity(suspected_text, ref)
            lex_sim = calculate_lexical_similarity(suspected_text, ref)
            avg_sim = (sem_sim + sem_sim_large + bert_sim) / 3
            results.append({
                'reference': ref[:100] + ("..." if len(ref) > 100 else ""),
                'semantic_similarity': sem_sim,
                'semantic_similarity_large': sem_sim_large,
                'bert_similarity': bert_sim,
                'lexical_similarity': lex_sim,
                'average_similarity': avg_sim,
                'plagiarism_detected': avg_sim > threshold or lex_sim > 0.9
            })
        return results
    if st.button("Check Plagiarism"):
        if not text.strip():
            st.warning("Please provide some text for analysis.")
        elif not reference_texts:
            st.warning("Please provide at least one reference text.")
        else:
            with st.spinner("Checking for plagiarism..."):
                results = plagiarism_check(text, reference_texts)
            for idx, res in enumerate(results):
                st.markdown(f"**Reference {idx+1}:**")
                st.markdown(f"- **Semantic Similarity:** {res['semantic_similarity']:.2f}")
                st.markdown(f"- **Semantic Similarity (Large):** {res['semantic_similarity_large']:.2f}")
                st.markdown(f"- **BERT Similarity:** {res['bert_similarity']:.2f}")
                st.markdown(f"- **Lexical Similarity:** {res['lexical_similarity']:.2f}")
                st.markdown(f"- **Average Similarity:** {res['average_similarity']:.2f}")
                st.markdown(f"- **Plagiarism Detected:** {'Yes' if res['plagiarism_detected'] else 'No'}")

# --- CITATION ---
if feature == "Citation Assistance":
    card("""
    <h3>Citation Assistance</h3>
    <p>Generate APA/MLA citations from DOI, URL, or manual entry.</p>
    """)
    citation_mode = st.radio("Choose input type:", ["DOI", "URL", "Manual"], horizontal=True)
    citation = ""
    error = None
    if citation_mode == "DOI":
        doi = st.text_input("Enter DOI (e.g. 10.1038/nphys1170)")
        if st.button("Generate Citation", key="cite_doi"):
            if not doi.strip():
                st.warning("Please enter a DOI.")
            else:
                headers = {"Accept": "application/vnd.citationstyles.csl+json"}
                url = f"https://doi.org/{doi.strip()}"
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        authors = ", ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in data.get('author', [])])
                        year = data.get('issued', {}).get('date-parts', [[None]])[0][0]
                        title = data.get('title', '')
                        journal = data.get('container-title', '')
                        apa = f"{authors} ({year}). {title}. {journal}. https://doi.org/{doi.strip()}"
                        mla = f"{authors}. \"{title}.\" {journal}, {year}, https://doi.org/{doi.strip()}"
                        st.markdown(f"**APA:** {apa}")
                        st.markdown(f"**MLA:** {mla}")
                    else:
                        st.error("DOI not found or invalid.")
                except Exception as e:
                    st.error(f"Error fetching citation: {e}")
    elif citation_mode == "URL":
        url = st.text_input("Enter URL")
        if st.button("Generate Citation", key="cite_url"):
            if not url.strip():
                st.warning("Please enter a URL.")
            else:
                doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", url, re.I)
                if doi_match:
                    doi = doi_match.group(0)
                    st.info(f"DOI detected: {doi}")
                    headers = {"Accept": "application/vnd.citationstyles.csl+json"}
                    url2 = f"https://doi.org/{doi}"
                    try:
                        resp = requests.get(url2, headers=headers, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            authors = ", ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in data.get('author', [])])
                            year = data.get('issued', {}).get('date-parts', [[None]])[0][0]
                            title = data.get('title', '')
                            journal = data.get('container-title', '')
                            apa = f"{authors} ({year}). {title}. {journal}. https://doi.org/{doi}"
                            mla = f"{authors}. \"{title}.\" {journal}, {year}, https://doi.org/{doi}"
                            st.markdown(f"**APA:** {apa}")
                            st.markdown(f"**MLA:** {mla}")
                        else:
                            st.error("DOI not found or invalid.")
                    except Exception as e:
                        st.error(f"Error fetching citation: {e}")
                else:
                    st.warning("No DOI found in URL. Please use manual entry or provide a DOI.")
    else:  # Manual
        author = st.text_input("Author(s) (e.g. Smith, John; Doe, Jane)")
        year = st.text_input("Year (e.g. 2020)")
        title = st.text_input("Title")
        journal = st.text_input("Journal/Source")
        url = st.text_input("URL (optional)")
        if st.button("Generate Citation", key="cite_manual"):
            if not (author and year and title and journal):
                st.warning("Please fill in all required fields.")
            else:
                apa = f"{author} ({year}). {title}. {journal}. {url}" if url else f"{author} ({year}). {title}. {journal}."
                mla = f"{author}. \"{title}.\" {journal}, {year}{', ' + url if url else ''}"
                st.markdown(f"**APA:** {apa}")
                st.markdown(f"**MLA:** {mla}") 