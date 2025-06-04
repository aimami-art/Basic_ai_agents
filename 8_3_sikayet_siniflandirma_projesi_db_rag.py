"""
kullanicidan gelen şikayetlerin analiz edilmesi, kategorize edilmesi, 
ilgili birime yönlendirilmesi ve uygun yanıt verilmesi sistemi gerceklestirelim.

+ otomatik mail (simulasyon) + raporlama (.excel)
+ RAG with db
+ memory 

"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from dotenv import load_dotenv
import os, warnings, sqlite3

warnings.filterwarnings("ignore")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def dbden_veri_al(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT baslik, icerik FROM urun_bilgileri")
    rows = cursor.fetchall()
    conn.close()
    return [Document(page_content=f"{baslik}:{icerik}") for baslik, icerik in rows]

# --- LLM & Embedding modeli
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.2)
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

# --- Memory eklendi
memory = ConversationBufferMemory(return_messages=True)

# --- 1. Ürün Bilgilerini yükle & FAISS index oluştur
 
documents = dbden_veri_al("urun_bilgileri.db")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# --- 2. Retrieval destekli QA zinciri (RAG)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
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

# --- 4. Prompt Template (memory geçmişi dahil)
prompt_template = PromptTemplate(
    input_variables=["sikayet", "bilgi", "gecmis"],
    partial_variables={"format_instructions": format_instructions},
    template="""
Aşağıda geçmiş konuşmalar yer almakta: geçmişi değerlendirerek de cevap ver, geçmiş sorulursa buradan cevap ver.

{gecmis}

Kullanıcıdan yeni bir şikayet geldi. Şikayeti analiz et (kategori, yönlendirme, kısa yanıt).
Ayrıca ürün bilgisi aşağıdadır:

{bilgi}

{format_instructions}

Şikayet: {sikayet}
"""
)

# --- Sonsuz döngü: çoklu şikayet
while True:
    sikayet = input("\n🗣 Şikayetinizi yazınız (Çıkmak için 'q'): ")
    if sikayet.lower() in ['q', 'exit', 'çık']:
        break

    # --- RAG bilgisi al
    bilgi = rag_chain.run(sikayet)

    # --- Geçmişi al (formatla)
    # sohbet_gecmisi = memory.buffer BU ŞEKİLDE YAZARSAN TÜM GEÇMİŞİ (HUMAN,AI,..) BİRLEŞTİRİP STR ŞEKLİNDE VERİR VE m.type GİBİ KOD YAZAMAZSIN
    sohbet_gecmisi = memory.chat_memory.messages
    gecmis = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in sohbet_gecmisi])

    print("gecmis: ", gecmis)
    # --- Prompt oluştur
    prompt = prompt_template.format(sikayet=sikayet, bilgi=bilgi, gecmis=gecmis)
    response = llm.invoke(prompt)
    result = parser.parse(response.content)

    # --- Belleğe kaydet
    memory.chat_memory.add_user_message(sikayet)
    memory.chat_memory.add_ai_message(response.content)

    # --- Sonuçları göster
    print("\n--- Şikayet Analizi ---")
    print("Kategori:", result["kategori"])
    print("Yönlendirme:", result["yonlendirme"])
    print("Cevap:", result["cevap"])

    # --- E-posta simülasyonu
    print("\n📬 Otomatik E-posta")
    print("Kime: kullanici@example.com")
    print("Konu: Şikayetiniz Hakkında")
    print(result["cevap"])