import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -----------------------------
# FAQ DATA
# -----------------------------
faqs = {
    "What is CodeAlpha?":
        "CodeAlpha is a software development company that provides internship and technology learning opportunities.",

    "How can I apply for an internship?":
        "You can apply for available CodeAlpha internship programs through the official application process.",

    "What tasks are required in the internship?":
        "Interns are required to complete the assigned tasks mentioned in their internship instructions.",

    "How many tasks should I complete?":
        "You should complete the minimum number of tasks specified in your internship instructions.",

    "How do I submit my completed task?":
        "Upload your completed project to GitHub and submit the project details through the provided submission form.",

    "Do I need to post my project on LinkedIn?":
        "Yes. The internship instructions ask interns to post their project on LinkedIn with the GitHub repository link.",

    "What is GitHub?":
        "GitHub is a platform used to store, manage, and share software projects and source code.",

    "What is Python?":
        "Python is a popular programming language used for web development, data science, artificial intelligence, and automation.",

    "What is artificial intelligence?":
        "Artificial Intelligence is the field of creating computer systems that can perform tasks that normally require human intelligence.",

    "What is machine learning?":
        "Machine learning is a branch of AI where computers learn patterns from data to make predictions or decisions."
}


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]


# -----------------------------
# FIND BEST FAQ MATCH
# -----------------------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def get_answer(user_question):
    cleaned_question = preprocess(user_question)

    if not cleaned_question:
        return "Please enter a question."

    user_vector = vectorizer.transform([cleaned_question])

    scores = (user_vector @ question_vectors.T).toarray()[0]

    best_match_index = scores.argmax()
    best_score = scores[best_match_index]

    if best_score < 0.2:
        return "Sorry, I don't know the answer to that question."

    best_question = questions[best_match_index]
    return faqs[best_question]


# ==============================
# STREAMLIT CHATBOT INTERFACE
# ==============================

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 FAQ Chatbot")
st.write("Ask me a question about Python, AI, Machine Learning, GitHub and more!")

user_question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):
    answer = get_answer(user_question)
    st.success(answer)

st.markdown("---")
st.caption("CodeAlpha FAQ Chatbot")
    