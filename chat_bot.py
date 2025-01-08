import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import load_config

# Load the configuration
cfg = load_config()
os.environ['GOOGLE_API_KEY'] = cfg.PALLM_API

# Load FAISS vectorstore from existing database
def load_vector_store():
    return FAISS.load_local(cfg.DB_FAISS_PATH, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Function to get the conversational chain
def get_conversational_chain(vector_store):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

# Simulate user input for interaction
def user_input(conversation, user_question):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print("Human:", message.content)
        else:
            print("Bot:", message.content)

# Main entry point for the script
def main():
    print("Welcome to PDF Chatbot!")
    vector_store = load_vector_store()

    conversation = get_conversational_chain(vector_store)
    
    while True:
        user_question = input("Ask a question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break
        user_input(conversation, user_question)

if __name__ == "__main__":
    main()
