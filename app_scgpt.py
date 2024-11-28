import os
import subprocess
import torch
import streamlit as st
import spacy
from src.attribution.attrb import AttributionModule
from src.generator.generator import Generator
from src.generator import prompts
from src.hallucination.hallucination import SelfCheckGPT

# Streamlit Configuration
st.set_page_config(page_title="💬 Medical Agent")

# Load SpaCy model for sentence tokenization
nlp = spacy.load("en_core_web_sm")

# Input for user query
with st.sidebar:
    st.title("Medical Agent")
    st.write("This is a chatbot that can answer medical queries from discharge summaries.")

## General Configuration
ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
EMBEDDINGS_FILE = os.path.join(ROOT_DIR, "embeddings", "paragraph_embeddings.npz")
PASSAGE_FILE = os.path.join(ROOT_DIR, "data", "passages.json")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# chat_model = "meta-llama/Llama-2-7b-chat-hf" 
chat_model="meta-llama/Meta-Llama-3-8B-Instruct"
attribution_model = "meta-llama/Llama-2-7b-chat-hf" 
# hallucination_model = "meta-llama/Llama-2-7b-chat-hf"
hallucination_model = "HPAI-BSC/Llama3-Aloe-8B-Alpha"

@st.cache_resource
def load_attribution_module():
    return AttributionModule(device="cuda:1")

@st.cache_resource
def load_generator():
    return Generator(model_path=chat_model, device="cuda:3")

@st.cache_resource
def load_hallucination_checker():
    return SelfCheckGPT(device="cuda:1", model=hallucination_model)

def split_sentences(text):
    # Use SpaCy to split the text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def get_response(user_query):
    # Generate response
    prompt = prompts.medical_prompt.format(joint_passages, user_query)
    answer = generator.generate_response(prompt)
    responses = generator.generate_response([prompt] * 5)
    
    # Attribution
    attribution_query = f"Question: {user_query}\nAnswer: {answer}"
    # Retrieve relevant paragraphs - Attribution Module
    retrieval_results = attribution_module.retrieve_paragraphs([attribution_query], search_index, paragraphs, k=1)
    print("\n\nRetrieval Results:\n\n", retrieval_results)
    retrieved_passages = retrieval_results[0]['retrieved_paragraphs'][0]

    # Check for hallucination
    hallucination_probability = hallucination_checker.hallucination_prop(answer, Passages=responses)

    # Split response into sentences using SpaCy
    sentences = split_sentences(answer)
    
    # Compile the response with hallucination probabilities
    response_with_hallucination = []
    for sentence, prob in zip(sentences, hallucination_probability):
        if prob is not None:
            sentence_with_prob = f"{sentence} <span style='background-color: yellow;'>{prob * 100:.2f}%</span>"
        else:
            sentence_with_prob = sentence
        response_with_hallucination.append(sentence_with_prob)
    
    response_with_hallucination_text = " ".join(response_with_hallucination)

    # Compile the response message
    assistant_message_parts = [
        f"**Generated Response:** {response_with_hallucination_text}",
        f"**Attribution:** {retrieved_passages}"
    ]
    assistant_message = "\n\n".join(assistant_message_parts)
    return assistant_message

# Initialize modules
attribution_module = load_attribution_module()
generator = load_generator()
hallucination_checker = load_hallucination_checker()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

## Processing Document
passages_file = os.path.join(ROOT_DIR, "data", "passages.json")
passages = attribution_module.load_paragraphs(passages_file=passages_file)
joint_passages = "\n".join(passages)

embedding_file_path = os.path.join(attribution_module.output_dir, "paragraph_embeddings.npz")
if not os.path.exists(embedding_file_path):
    attribution_module.vectorize_paragraphs(passages_file, embedding_file_path)

search_index, paragraphs = attribution_module.create_faiss_index(embedding_file_path=embedding_file_path, ngpu=1)

# Add a clear history button
def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# Input for user query
user_query = st.chat_input("Enter your medical query:")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

# Generate a new response if the last message is not from the assistant
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
            
            assistant_message = get_response(st.session_state.chat_history[-1]["content"])
            placeholder.markdown(assistant_message, unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
