from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.tools import tool

from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

@tool
def toplama_araci(input: str) -> str:
    """
    Bu araç, verilen sayıları toplar.
    Örnek kullanım: 5 ve 8
    """
    try:
        a,b = [int(x.strip()) for x in input.split("ve")]
        return f"Toplam {a} ve {b} sayilarinin toplami: {a + b}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen iki sayiyi '5 ve 8 ' formatinda giriniz."
    
@tool
def bolme_araci(input: str) -> str:
    """
    Bu araç, verilen sayıları böler.
    Örnek kullanım: 10 ve 2
    """
    try:
        a,b = [int(x.strip()) for x in input.split("ve")]
        if b == 0:
            return "Hata: Bir sayi sifira bölünemez."
        return f"Bölüm {a} ve {b} sayilarinin bölümü: {a / b}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen iki sayiyi '10 ve 2' formatinda giriniz."
    
tools = [toplama_araci, bolme_araci]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

soru = "10 sayyisini 2 ye böl, sonrasinda bölümü 5 ile topla."

response = agent.run(soru)

print(f"Cevap : {response}")