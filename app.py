import streamlit as st
import pickle as pk
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer

# Load necessary files
tfidf = pk.load(open('./Pickle_files/vectorizer.pkl', 'rb'))
model = pk.load(open('./Pickle_files/model.pkl', 'rb'))

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Text transformation function
def transform_text(text):
    lower_text = text.lower()
    tokenized_text = nltk.word_tokenize(lower_text)
    alnum_words = [word for word in tokenized_text if word.isalnum()]
    filtered_words = [
        word for word in alnum_words
        if word not in stopwords.words('english') and word not in string.punctuation
    ]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return " ".join(stemmed_words)

# Page configuration
st.set_page_config(
    page_title="Email/SMS Spam Predictor",
    page_icon="📧",
    layout="centered"
)

# Dynamic theming with prefers-color-scheme
st.markdown(
    """
    <style>
    /* General container styling */
    .main-container {
        margin-top: 20px;
        padding: 20px;
        border-radius: 8px;
    }

    /* Light mode styling */
    @media (prefers-color-scheme: light) {
        body {
            background-color: #f9f9f9;
            color: #333;
        }
        .header-title {
            color: #333;
        }
        .result-box {
            background-color: #e9ecef;
            border: 1px solid #ddd;
        }
        .ham-result {
            color: #155724;
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .spam-result {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
        }
        .stButton button:hover {
            background-color: #0056b3;
            color: white;
        }
    }

    /* Dark mode styling */
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #121212;
            color: #f1f1f1;
        }
        .header-title {
            color: #f1f1f1;
        }
        .result-box {
            background-color: #333;
            border: 1px solid #555;
        }
        .ham-result {
            color: #d4edda;
            background-color: #155724;
            border-color: #28a745;
        }
        .spam-result {
            color: #f8d7da;
            background-color: #721c24;
            border-color: #dc3545;
        }
        .stButton button {
            background-color: #1e88e5;
            color: white;
        }
        .stButton button:hover {
            background-color: #1565c0;
            color: white;
        }
    }

    /* Header styling */
    .header-title {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Subheader styling */
    .subheader {
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Input box styling */
    .stTextArea textarea {
        font-size: 1.1em;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #ddd;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main application layout
with st.container():
    st.markdown('<div class="header-title">📧 Email/SMS Spam Predictor 📧</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Check if a message is spam or not with accuracy and reliability.</div>', unsafe_allow_html=True)

    # Input section
    st.text_area("Enter your message below:", key="input_msg", placeholder="Type your email or SMS message here...")

    # Predict button
    if st.button("Check"):
        input_msg = st.session_state.input_msg
        if input_msg:
            with st.spinner("Analyzing the message..."):
                transformed_msg = transform_text(input_msg)
                vector_msg = tfidf.transform([transformed_msg])
                prediction = model.predict(vector_msg)

            # Display result
            if prediction == 0:
                st.markdown('<div class="result-box ham-result">This message is HAM (Not Spam).</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box spam-result">This message is SPAM.</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a message to analyze.")

# Footer
st.markdown(
    """
    <div class="footer">
        Built by <strong>Yash</strong> using <strong>Streamlit</strong> | © 2024
    </div>
    """,
    unsafe_allow_html=True
)
