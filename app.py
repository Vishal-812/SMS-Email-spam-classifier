import  streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc, #e0f2fe);
}

h1 {
    text-align: center;
    color: #1e293b;
}

textarea {
    border-radius: 12px !important;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Classifier")

## input_sms = st.text_area("Enter your message")
input_sms = st.text_area(
    "📩 Enter Your Message",
    placeholder="Type your SMS or email here..."
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button("🔎 Predict", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.rerun()

if predict:
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3. model
    result = model.predict(vector_input.toarray())

    # 4. display
    if result[0] == 1:
        st.error("🚨 SPAM MESSAGE")

    else:
        st.success("✅ HAM MESSAGE")






