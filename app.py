# app.py
# AI Travel Assistant Chatbot (Streamlit + LangChain + HuggingFace + FAISS)
# Fixed version: ensures natural text answers, avoids numeric outputs.

import traceback

try:
    import streamlit as st
except Exception:
    print("Missing dependency: streamlit. Run: pip install streamlit")
    raise

st.set_page_config(page_title="AI Travel Assistant", page_icon="🌍", layout="centered")

st.title("🌍 AI Travel Assistant")
st.markdown("Ask me about *destinations, flights, hotels, food, visas, or travel tips*.")

# ---------------- Dependency checks ----------------
missing_pkgs = []

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
except Exception:
    missing_pkgs.append("langchain-community")

try:
    import langchain
except Exception:
    missing_pkgs.append("langchain")

try:
    from transformers import pipeline
except Exception:
    missing_pkgs.append("transformers")

try:
    import sentence_transformers  # noqa: F401
except Exception:
    missing_pkgs.append("sentence-transformers")

try:
    import faiss  # noqa: F401
except Exception:
    missing_pkgs.append("faiss-cpu")

try:
    import torch  # noqa: F401
except Exception:
    missing_pkgs.append("torch")

if missing_pkgs:
    st.error("Some Python packages required by this app are not installed.")
    st.write("Missing packages detected: " + ", ".join(missing_pkgs))
    st.write("To install them, run:")
    st.code("pip install " + " ".join(missing_pkgs))
    st.stop()

# ---------------- Travel FAQ Data ----------------
faq_data = [
    {"question": "Best time to visit Europe?",
     "answer": "Spring (April–June) and fall (September–October) are the best times to visit Europe for pleasant weather and fewer crowds."},

    {"question": "Do I need a visa for Schengen countries?",
     "answer": "Most non-EU travelers require a Schengen visa to visit countries like France, Germany, Italy, and Spain. Check official embassy websites."},

    {"question": "What are budget-friendly destinations?",
     "answer": "Popular budget-friendly destinations include Thailand, Vietnam, Portugal, and Bali, offering affordable food, stays, and transport."},

    {"question": "How to find cheap flights?",
     "answer": "Use flight comparison sites like Skyscanner, Google Flights, and Kayak. Book 6–8 weeks in advance and be flexible with dates."},

    {"question": "What are the safest travel destinations?",
     "answer": "Switzerland, Japan, New Zealand, and Canada consistently rank among the safest countries for travelers."},

    {"question": "Best honeymoon destinations?",
     "answer": "Top honeymoon picks include Maldives, Bora Bora, Santorini (Greece), Switzerland, and Bali."},

    {"question": "How to avoid jet lag?",
     "answer": "Adjust your sleep schedule before flying, stay hydrated, avoid alcohol, and get sunlight at your destination."},

    {"question": "What are must-have travel apps?",
     "answer": "Must-have apps: Google Maps, Skyscanner, Airbnb, TripAdvisor, Google Translate, and XE Currency."},

    {"question": "Do I need travel insurance?",
     "answer": "Yes, travel insurance is highly recommended. It covers medical emergencies, trip cancellations, lost luggage, and delays."},

    {"question": "What are some famous world festivals?",
     "answer": "Famous festivals include Rio Carnival (Brazil), Oktoberfest (Germany), Holi (India), and La Tomatina (Spain)."}
]

# ---------------- Prepare Documents ----------------
from langchain.docstore.document import Document
docs = [Document(page_content=f["answer"], metadata={"question": f["question"]}) for f in faq_data]

# ---------------- Embeddings ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ---------------- FAISS Vector Store ----------------
try:
    vectorstore = FAISS.from_documents(docs, embeddings)
except Exception:
    st.error("Failed to create FAISS vectorstore. Likely faiss installation issue.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- QA Model ----------------
try:
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
except Exception:
    st.error("Failed to initialize HuggingFace pipeline. Check transformers + torch install.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- Custom Prompt ----------------
prompt_template = """
You are an AI Travel Assistant. 
Answer the question based on the following context in a clear, complete sentence.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# ---------------- QA Chain ----------------
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
except Exception:
    st.error("Failed to build the RetrievalQA chain.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- Streamlit UI ----------------
st.sidebar.title("📌 Quick Travel FAQs")
quick_links = [
    "Best time to visit Europe",
    "Visa requirements for Schengen",
    "Cheap flight tips",
    "Honeymoon destinations",
    "Travel insurance importance",
    "Famous festivals"
]
for item in quick_links:
    st.sidebar.markdown(f"- {item}")

query = st.text_input("💬 Ask your travel question:")

if st.button("Get Answer") and query:
    with st.spinner("Searching for the best travel advice..."):
        try:
            result = qa_chain(query)
        except Exception:
            st.error("Error while running the QA chain.")
            st.code(traceback.format_exc())
            st.stop()

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

    if source_docs:
        matched_question = source_docs[0].metadata.get("question", "")
        st.success(f"*Answer:* {answer}")
        st.info(f"📌 Based on: {matched_question}")
    else:
        st.warning("Sorry, I couldn't find an answer. Try rephrasing your question.")
