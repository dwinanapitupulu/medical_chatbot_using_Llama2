from src.helper import load_pdf,text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

#print(PINECONE_API_KEY)
#print(PINECONE_API_ENV)

extracted_data = load_pdf("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "mymedical-chatbot"
    # Now do stuff
if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

#Creating Embeddings for Each of The Text Chunks & storing
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)