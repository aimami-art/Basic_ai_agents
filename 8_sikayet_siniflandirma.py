"""
kullanicidan gelen şikayetlerin analiz edilmesi, kategorize edilmesi, 
ilgili birime yönlendirilmesi ve uygun yanıt verilmesi sistemi gerceklestirelim.

"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

# .env dosyasını yükle
load_dotenv()
# OpenAI API anahtarını al
api_key = os.getenv("OPENAI_API_KEY")

# --- 1. LLM Modeli ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=api_key,
    temperature=0.2 # Düşük sıcaklık, daha kararlı ve tutarlı cevaplar için
)

# --- 2. Beklenen JSON formatı için şema tanımları ---
response_schemas = [
    ResponseSchema(name="kategori", description="Şikayet hangi kategoriye ait? (kargo, iade, ödeme, teknik, iletişim)"),
    ResponseSchema(name="yonlendirme", description="Hangi departmana yönlendirilmeli?"),
    ResponseSchema(name="cevap", description="Kullanıcıya uygun tathmin edici bir yanıt metni")
]

# Parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# --- 3. Prompt Template ---
template = PromptTemplate(
    input_variables=["sikayet"],
    partial_variables={"format_instructions": format_instructions},
    template="""
Bir kullanıcı şikayeti aldın. 
Bu şikayeti anlamlandır, uygun kategoriyi belirle, 
yönlendirme yap ve kullanıcıya kısa bir açıklama yaz.

{format_instructions}

Şikayet: {sikayet}
"""
)

# --- 4. Kullanıcı girdisi (örnek) ---
sikayet = "2 gündür kargom gelmedi, müşteri hizmetlerine de ulaşamıyorum."
# sikayet = "Uygulamaya giriş yapamıyorum, sürekli hata veriyor ve şifre sıfırlama bağlantısı da çalışmıyor."
sikayet = "ürün tarif edildiği gibi değil ve hasarlı geldi, iade etmek istiyorum."

# Prompt'u oluştur
prompt = template.format(sikayet=sikayet)

# LLM'den yanıt al
response = llm.invoke(prompt)

# Yapılandırılmış veriyi ayrıştır
result = parser.parse(response.content)

# Sonucu göster
print("\n--- Şikayet Analizi ---")
print("Kategori:", result["kategori"])
print("Yönlendirme:", result["yonlendirme"])
print("Cevap:", result["cevap"])

