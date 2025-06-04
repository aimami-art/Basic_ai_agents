from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=openai_api_key
)

# format_instructions oluşturma :

response_schemas = [
    ResponseSchema(name="urun_adi", description="Ürünün adı"),
    ResponseSchema(name="kategori", description="Ürün hangi kategoriye ait"),
    ResponseSchema(name="fiyat", description="Tahmini fiyat (TL olarak)"),
    ResponseSchema(name="stok_durumu", description="Stokta var mı yok mu")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = parser.get_format_instructions()

# soruyu şekillendirme :

tamplate = PromptTemplate(
    input_variables=["soru"],
    partial_variables={"format_instructions": format_instructions},
    template="""
Aşağıdaki soruya cevap ver. Cevabın aşağıda belirtilen formatta olmasına dikkat et.

{format_instructions}

Soru: {soru}
"""
)

soru = "Samsung Galaxy S23 hakkında bilgi ver"

# CEVAP ALMA :

prompt = tamplate.format(soru=soru)

response = llm.invoke(prompt)

structured_response = parser.parse(response.content) # response_schemas a uygun olarak sözlüğe çevirir.

print("Cevap:", structured_response)