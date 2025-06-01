"""
kullanicidan gelen şikayetlerin analiz edilmesi, kategorize edilmesi, 
ilgili birime yönlendirilmesi ve uygun yanıt verilmesi sistemi gerceklestirelim.

+ otomatik mail (simulasyon) + raporlama (.csv)
+ RAG

"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# --- LLM & Embedding modeli
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.2)
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

# --- 1. Ürün Bilgilerini yükle & FAISS index oluştur
loader = TextLoader("rag_veriler.txt")
documents = loader.load()
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# --- 2. Retrieval destekli QA zinciri (RAG)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # basic RAG zinciri
    verbose=False
)

# --- 3. Şikayet şeması
schemas = [
    ResponseSchema(name="kategori", description="kargo, iade, ödeme, teknik, iletişim"),
    ResponseSchema(name="yonlendirme", description="İlgili departman"),
    ResponseSchema(name="cevap", description="Kısa kullanıcı yanıtı"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

prompt_template = PromptTemplate(
    input_variables=["sikayet", "bilgi"],
    partial_variables={"format_instructions": format_instructions},
    template="""
Kullanıcıdan bir şikayet geldi. Önce şikayeti analiz et (kategori, yönlendirme, kısa yanıt).
Şikayetle ilgili aşağıdaki bilgiler sana destek olabilir:

{bilgi}

{format_instructions}

Şikayet: {sikayet}
"""
)

# --- 4. Şikayet al
sikayet = input("Şikayet metnini yazınız: ")

# --- 5. RAG ile ürün bilgisi getir
bilgi = rag_chain.run(sikayet)

# --- 6. Prompt'u oluştur ve cevapla
prompt = prompt_template.format(sikayet=sikayet, bilgi=bilgi)
response = llm.invoke(prompt)
result = parser.parse(response.content)

# --- 7. Sonucu göster
print("\n--- Şikayet Analizi ---")
print("Kategori:", result["kategori"])
print("Yönlendirme:", result["yonlendirme"])
print("Cevap:", result["cevap"])

# --- 8. E-posta simülasyonu
print("\n--- 📬 Otomatik E-posta ---")
print("Kime: kullanici@example.com")
print("Konu: Şikayetiniz Hakkında")
print(result["cevap"])