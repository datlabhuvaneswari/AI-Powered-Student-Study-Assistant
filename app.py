import streamlit as st
import chromadb
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os

 

st.set_page_config(
    page_title="AI Student Study Assistant",
    page_icon="🎓",
    layout="wide"
)

USERS = {
    "student": "1234",
    "admin": "admin123",
    "bhuvaneswari": "12345"
}


def login_page():

    st.title("🔐 AI Student Study Assistant")

    st.subheader("Login to continue")

    username = st.text_input(
        "👤 Username",
        placeholder="Enter username"
    )

    password = st.text_input(
        "🔑 Password",
        type="password",
        placeholder="Enter password"
    )

    login_button = st.button(
        "🚀 Login",
        use_container_width=True
    )

    if login_button:

        if username in USERS and USERS[username] == password:

            st.session_state["logged_in"] = True
            st.session_state["username"] = username

            st.success("✅ Login successful!")

            st.rerun()

        else:

            st.error(
                "❌ Invalid username or password."
            )

    st.markdown("---")

    st.info(
        "Demo Login\n\n"
        "Username: student\n\n"
        "Password: 1234"
    )


 

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


if not st.session_state["logged_in"]:

    login_page()

    st.stop()


 
st.sidebar.success(
    f"👋 Welcome, {st.session_state['username']}!"
)

if st.sidebar.button("🚪 Logout"):

    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

    st.rerun()


 

st.title("🎓 AI-Powered Student Study Assistant")

st.write(
    "Upload your study material and use AI to summarize, "
    "ask questions, generate quizzes, and create study notes."
)

 

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

 

@st.cache_resource
def create_chroma_client():
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="student_documents"
    )

    return collection


collection = create_chroma_client()

 

OLLAMA_URL = "http://localhost:11434/api/generate"

 
OLLAMA_MODEL = "llama3.2"


 
def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response generated.")

        return f"Ollama error: {response.text}"

    except requests.exceptions.ConnectionError:
        return (
            "❌ Could not connect to Ollama.\n\n"
            "Please make sure Ollama is running."
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"


 

def extract_pdf_text(uploaded_file):
    text = ""

    try:
        reader = PdfReader(uploaded_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

    return text


 

def split_text(text, chunk_size=800):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


 

def store_document(chunks):
    try:
         
        embeddings = embedding_model.encode(chunks).tolist()

         
        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

        return True

    except Exception as e:
        st.error(f"Error storing document: {e}")
        return False


 

def search_document(question, number_of_results=4):
    try:
        question_embedding = embedding_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=number_of_results
        )

        documents = results.get("documents", [[]])[0]

        return documents

    except Exception as e:
        st.error(f"Search error: {e}")
        return []


 

def answer_question(question):
    documents = search_document(question)

    if not documents:
        return "No relevant information found in the uploaded document."

    context = "\n\n".join(documents)

    prompt = f"""
You are an AI Student Study Assistant.

Answer the student's question using ONLY the information
provided in the study material below.

If the answer is not available in the material,
say that the answer is not available in the uploaded document.

Study Material:
----------------
{context}
----------------

Student Question:
{question}

Give a clear and easy-to-understand answer.
"""

    return ask_ollama(prompt)


 

st.sidebar.header("📚 Study Tools")

tool = st.sidebar.radio(
    "Choose a tool:",
    [
        "Ask Questions",
        "Summarize Notes",
        "Generate Quiz",
        "Create Study Notes"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a PDF first, then select a study tool."
)


 

uploaded_file = st.file_uploader(
    "📄 Upload your study material (PDF)",
    type=["pdf"]
)


 

if uploaded_file is not None:

    if st.button("📥 Process Study Material"):

        with st.spinner("Reading PDF..."):

            text = extract_pdf_text(uploaded_file)

        if not text.strip():

            st.error(
                "❌ Could not extract text from this PDF."
            )

        else:

            chunks = split_text(text)

            with st.spinner("Creating embeddings and storing document..."):

                success = store_document(chunks)

            if success:

                st.success(
                    f"✅ PDF processed successfully! "
                    f"{len(chunks)} chunks stored."
                )

                st.session_state["document_text"] = text
                st.session_state["document_processed"] = True


 

document_ready = st.session_state.get(
    "document_processed",
    False
)


 

if tool == "Ask Questions":

    st.header("💬 Ask Questions")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is normalization?"
    )

    if st.button("🔍 Get Answer"):

        if not document_ready:

            st.warning(
                "⚠️ Please upload and process a PDF first."
            )

        elif not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner("Finding answer..."):

                answer = answer_question(question)

            st.subheader("🤖 Answer")

            st.write(answer)


 

elif tool == "Summarize Notes":

    st.header("📝 Summarize Notes")

    if not document_ready:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

    else:

        if st.button("✨ Generate Summary"):

            text = st.session_state["document_text"]

             
            text_for_summary = text[:15000]

            prompt = f"""
You are a student study assistant.

Summarize the following study material.

Make the summary:
- Easy to understand
- Short
- Exam-friendly
- Organized with headings
- Use bullet points where useful

Study Material:
----------------
{text_for_summary}
----------------
"""

            with st.spinner("Generating summary..."):

                summary = ask_ollama(prompt)

            st.subheader("📋 Summary")

            st.write(summary)


 

elif tool == "Generate Quiz":

    st.header("🧠 Generate Quiz")

    number = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5
    )

    if not document_ready:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

    else:

        if st.button("🎯 Generate Quiz"):

            text = st.session_state["document_text"]

            text_for_quiz = text[:15000]

            prompt = f"""
You are an AI teacher.

Create {number} multiple-choice questions
from the study material below.

For every question provide:

Question:
A)
B)
C)
D)

Correct Answer:
Explanation:

Make the questions useful for exam preparation.

Study Material:
----------------
{text_for_quiz}
----------------
"""

            with st.spinner("Generating quiz..."):

                quiz = ask_ollama(prompt)

            st.subheader("🎯 Practice Quiz")

            st.write(quiz)


 

elif tool == "Create Study Notes":

    st.header("📚 Create Study Notes")

    if not document_ready:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

    else:

        if st.button("📖 Create Notes"):

            text = st.session_state["document_text"]

            text_for_notes = text[:15000]

            prompt = f"""
You are an expert study assistant.

Convert the following study material into
well-organized student study notes.

Use this format:

1. Important Concepts
2. Definitions
3. Key Points
4. Examples
5. Important Exam Points
6. Quick Revision

Keep the language simple and clear.

Study Material:
----------------
{text_for_notes}
----------------
"""

            with st.spinner("Creating study notes..."):

                notes = ask_ollama(prompt)

            st.subheader("📖 Study Notes")

            st.write(notes)


 

st.markdown("---")

st.caption(
    "🎓 AI-Powered Student Study Assistant | "
    "Streamlit + ChromaDB + Sentence Transformers + Ollama"
)
