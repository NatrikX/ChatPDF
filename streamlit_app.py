import os
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setting page title and header
st.set_page_config(page_title="ChatPDF", page_icon="💬")
st.markdown("<h1 style='text-align: center;'>ChatPDF 💬</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Transforming PDFs into Dynamic Chat Experience 🚀</h5>", unsafe_allow_html=True)



# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]
# if 'model_name' not in st.session_state:
#     st.session_state['model_name'] = []
# if 'cost' not in st.session_state:
#     st.session_state['cost'] = []
# if 'total_tokens' not in st.session_state:
#     st.session_state['total_tokens'] = []
# if 'total_cost' not in st.session_state:
#     st.session_state['total_cost'] = 0.0

@st.cache_resource
def build_model(filename):
    print("inside build model")
    text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 2000,
            chunk_overlap  = 100,
            length_function = len,
        )
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split(text_splitter=text_splitter)
    print("documenting index")
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
        )
    print("building conversational chain")
    qa = ConversationalRetrievalChain.from_llm(
          ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=712, temperature=0.5, openai_api_key=openai_api_key), 
          faiss_index.as_retriever(), 
          memory=memory,
          return_source_documents=True
          )
    return qa

# generate a response
def generate_response(prompt):
    result = model({"question": prompt})
    return result['answer']

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation

with st.sidebar:
    st.title("ChatPDF 💬")
    st.markdown("Transforming PDFs into Dynamic Chat Experience")
    st.markdown("---")
    uploaded_file = st.file_uploader(label="Upload PDF")
    st.markdown("---")
    openai_api_key = st.text_input("API Key*", key="openai_api_key", type="password")
    # st.markdown("<p style='text-align: center; font-size:12px; color:#A9A9A9'>@AuditAdvisor is a personal project developed to provide guidance on general audit queries. This application utilizes OpenAI's GPT-3 model to process and respond to user requests</p>", unsafe_allow_html=True)


if uploaded_file is not None and openai_api_key is not None and openai_api_key != "":
    with open(os.path.join(uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())

    model = build_model(uploaded_file.name)

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
