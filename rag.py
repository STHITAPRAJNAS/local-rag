from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    memory = None

    def __init__(self):
        self.model = ChatOllama(model="llama3:latest", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True, output_key="answer")
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. Use only the following pieces of retrieved context 
            to answer the question. If the context doesn't contain the answer, 
            say "I don't have enough information to answer that question." 
            Do not use any prior knowledge. Keep your answer concise, using no more than five sentences.
            You can answer greeting type questions 
            
            Context: {context}
            Chat History: {chat_history}
            Human: {question}
            Assistant:
            """
        )

    def ingest(self, pdf_path):
        docs = PyPDFLoader(file_path=pdf_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': 3,
                'score_threshold': 0.5
            },
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=False,
            verbose=True
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please ingest a PDF file first."

        result = self.chain({"question": query})
        answer = result["answer"]
        #sources = [doc.metadata.get('source', 'Unknown') for doc in result.get('source_documents', [])]

        return f"Answer: {answer}\n"

    def clear(self):
        if self.vector_store:
            self.vector_store.delete_collection()
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.memory.clear()
