"""
plan and execute multiple steps in a single agent call
1. plan yap (plan)
2. adım adım uygula (execute)

senaryo: bir öğrencinin sınav notunu hesapla sonucu degerlendir

1) plan: sınavları topla, ortalamayı hesapla, sonucu değerlendir(yorum yap)
2) execute: plani toolar ile adim adim uygula

notlari_topla_tool
ortalama_hesapla_tool
degerlendir_tool

"""

from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.tools import tool
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=api_key,
)

@tool
def notlari_topla_tool(input: str) -> str:
    """Öğrencinin sınav notlarını toplayan araç. Örn: '80, 90, 75'"""
    try:
        notlar = [int(x.strip()) for x in input.split(",")]
        toplam = sum(notlar)
        return f"Toplam not: {toplam}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen notları '80, 90, 75' formatında giriniz."

@tool
def ortalama_hesapla_tool(input: str) -> str:
    """Öğrencinin not ortalamasını hesaplayan araç. Örn: '80, 90, 75'"""
    try:
        notlar = [int(x.strip()) for x in input.split(",")]
        ortalama = sum(notlar) / len(notlar)
        return f"Not ortalaması: {ortalama}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen notları '80, 90, 75' formatında giriniz."

@tool
def degerlendir_tool(input: str) -> str:
    """Öğrencinin not ortalamasını değerlendiren araç. Örn: '80'"""
    try:
        ortalama = float(input.strip())
        if ortalama >= 85:
            return "Başarılı"
        elif ortalama >= 70:
            return "Orta"
        else:
            return "Başarısız"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen not ortalamasını '80' formatında giriniz."
    

tools = [
    notlari_topla_tool,
    ortalama_hesapla_tool,
    degerlendir_tool
]

planner = load_chat_planner(llm=llm)

executor = load_agent_executor(
    llm=llm,
    tools=tools,
    verbose=True
)

plan_and_execute = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True
)

hedef = "90, 90, 75 notlarım var. Sınav notumu hesapla ve sonucu değerlendir."
result = plan_and_execute.run(hedef)
print(f"Plan ve Execute Sonucu: {result}")