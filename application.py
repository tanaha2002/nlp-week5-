import streamlit as st
import joblib
import underthesea

st.title("Phân loại văn bản Tiếng Việt")

model = joblib.load("gnb_model.pkl")

prompt = st.chat_input("Say something")
if prompt:
    st.write("You: ", prompt)
    text = underthesea.word_tokenize(prompt, format="text")
    # print(text)
    tfidf = joblib.load("tfidf.pkl")    
    text = tfidf.transform([text])
    label = model.predict(text.toarray())[0]
    st.markdown(f'<p style="text-align: right;">Predict: {label}</p>', unsafe_allow_html=True)


#create a sidebar to upload text file
st.sidebar.title("Tải file ")
uploaded_file = st.sidebar.file_uploader("Chọn file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    text = bytes_data.decode("utf-8")
    st.write("You: ", text)
    text = underthesea.word_tokenize(text, format="text")
    # print(text)
    tfidf = joblib.load("tfidf.pkl")    
    text = tfidf.transform([text])
    label = model.predict(text.toarray())[0]
    st.markdown(f'<p style="text-align: right;">{label}</p>', unsafe_allow_html=True)