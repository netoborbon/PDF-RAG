from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from data_ingestion import *
from rag_models import *

class RAGChat:

    def  __init__(self, k = 5, session_id = None):
        # conversation llm model to be used
        self.model = llm_model()
        # embeddings to be imlemented
        self.embeddings = embedding_model()
        # vector space where the embeddings will be stored
        self.vector_space = vector_store_db(self.embeddings,session_id)
        # Top k that will be returned
        self.k = k
        # similarity space
        self.retriever = self.vector_space.as_retriever(search_kwargs={"k": k})
        # query history
        self.conversation_history = []
        # system prompt
        self.system_prompt = (
            "You are a strict document-grounded assistant.\n\n"

            "You must answer using ONLY the information explicitly present in the provided context.\n\n"

            "MANDATORY RULES:\n"
            "1. ONLY use information explicitly stated in the conctext.\n"
            "2. Do NOT use any external knowledge.\n"
            "3. Do NOT add explanations beyond what is in the context.\n"
            "4. Do NOT infer, assume, generalize, or extrapolate.\n"
            "5. Do NOT provide partial answers if the information is incomplete.\n"
            "6. Your entire response must be fully supported by the context.\n"
            "7. Do NOT include phrases like 'based on general knowledge' or similar, even if you are confident.\n"
            "8. Do NOT continue the answer after giving a fallback response.\n\n"

            "RESPONSE RULES:\n"
            "- If the answer is fully supported by the context, provide the answer.\n"
            "- If the context does NOT contain enough information, respond EXACTLY with:\n"
            "I don't have enough information in the provided documents to answer that.\n"
            "- If the question is unrelated to the context, respond EXACTLY with:\n"
            "That topic is not covered in the provided documents.\n\n"

            "Context:\n{context}"
        )
    
    # load the documents into the vector space
    def ingest(self,path):
        # call read pdf function
        docs = ingest_pdfs(path)
        # split the docs into chunks
        chunks = doc_chunks(docs)
        # add the chunks in the vector space
        self.vector_space.add_documents(documents=chunks)
        # relaod the retriever space
        self.retriever = self.vector_space.as_retriever(search_kwargs={"k": self.k})
        
    # format the information for llm ingestion
    def format_docs(self,docs):
        return "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )

    # make the question
    def ask(self,question):
        # Retrieve relevant docs
        retrieved_docs = self.retriever.invoke(question)
        # format the information for llm ingestion
        context = self.format_docs(retrieved_docs)

        # structure the message passed to the llm
        message = [
            SystemMessage(content=self.system_prompt.format(context=context)),
            *self.conversation_history,
            HumanMessage(content=question)
        ]

        # make the message
        response = self.model.invoke(message)

        # retain memory only of the last 2 questions
        if len(self.conversation_history) == 6:
            del self.conversation_history[:2]

        # add the question and the answer to history
        self.conversation_history.extend([
        HumanMessage(content=question),
        AIMessage(content=response.content)
        ])

        # return response
        return response.content