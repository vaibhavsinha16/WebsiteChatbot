from flask import Flask, render_template, request, jsonify
from website_chatbot import process_website, rag_pipeline, OpenAIEmbeddings, ChatOpenAI, FAISS, RetrievalQA, ConversationBufferMemory, PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables to store the QA chain and vector store
qa_chain = None
vectorstore = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_website', methods=['POST'])
def process_website_route():
    global qa_chain, vectorstore
    
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({'error': 'OpenAI API key not configured'}), 500
    
    try:
        # Process website
        texts = process_website(url)
        if not texts:
            return jsonify({'error': 'No content found on the website'}), 400
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Initialize QA chain
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.4)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        template = """Context: {context}
        Question: {question}
        Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
        But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?
        """
        
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return jsonify({'message': 'Website processed successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_chain, vectorstore
    
    if not qa_chain or not vectorstore:
        return jsonify({'error': 'Please process a website first'}), 400
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        response = rag_pipeline(question, qa_chain, vectorstore)
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
