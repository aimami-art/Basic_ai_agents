"""
agenta a short-term memory ekleyelim
onceki sorulari ve cevaplari hatirlayabilsin

ornek senaryo 
konusma 1: 5 ile 2 yi topla,
konusma 2: sonucu 2 ile carp

"""

from langchain.chat_models import ChatOpenAI  # chat tabanlı openai modeli
from langchain.agents import Tool, AgentExecutor, initialize_agent  # agent ve tool kullanımı için gerekli sınıflar
from langchain.tools import tool  # langchain araçlarını içe aktar
from langchain.memory import ConversationBufferWindowMemory  # konuşma belleği için gerekli sınıf

from dotenv import load_dotenv  # .env dosyasını yüklemek için
import os  # işletim sistemi ile etkileşim için
import warnings  # uyarıları yönetmek için  
warnings.filterwarnings("ignore")  # uyarıları görmemek için

# .env dosyasını yükle
load_dotenv()  # .env dosyasını yükle
openai_api_key = os.getenv("OPENAI_API_KEY")  # .env dosyasından OpenAI API anahtarını al

# OpenAI modelini tanımla
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # kullanılacak modelin adı
    temperature=0.7,  # cevabın çeşitliliği (0 = kararlı, 1 = rastgele)
    openai_api_key=openai_api_key  # API anahtarını ver
)

@tool
def toplama_araci(input: str) -> str:
    """Verilen iki sayıyı toplayan araç. orn 5 ve 8"""
    try:
        a,b = [int(x.strip()) for x in input.split("ve")]
        return f"Toplam {a} ve {b} = {a+b}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen iki sayıyı '5 ve 8' formatında giriniz."

@tool
def bolme_araci(input: str) -> str:
    """Verilen iki sayıyı bölen araç. orn 10 ve 2"""
    try:
        a, b = [int(x.strip()) for x in input.split("ve")]
        if b == 0:
            return "Hata: Sıfıra bölme hatası."
        return f"Bölüm {a} ve {b} = {a / b}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen iki sayıyı '10 ve 2' formatında giriniz."

@tool
def carpma_araci(input: str) -> str:
    """Verilen iki sayıyı çarpan araç. orn 3 ve 4"""
    try:
        a, b = [int(x.strip()) for x in input.split("ve")]
        return f"Çarpım {a} ve {b} = {a * b}"
    except Exception as e:
        return f"Hata: {str(e)}. Lütfen iki sayıyı '3 ve 4' formatında giriniz."

tools = [toplama_araci, bolme_araci, carpma_araci]

# memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # bellek anahtarı
    return_messages=True,  # mesajları döndür
    output_key="output",  # çıktı anahtarı
    k = 5  # en son 5 mesajı hatırla

)

agent = initialize_agent(
    tools=tools,  # kullanılacak araçlar
    llm=llm,  # OpenAI modeli
    agent_type="zero-shot-react-description",  # agent tipi
    verbose=True,  # ayrıntılı çıktı
    memory=memory  # bellek ekle
)

print("Yapay zeka ile matematik işlemleri yapabilirsiniz.")
while True:

    soru = input("Soru: ")

    try:
        if soru.lower() in ["exit", "quit", "çıkış"]:
            print("Çıkılıyor...")
            break

        chat_gecmisi = memory.chat_memory.messages
        son_ai_cevabi = ""
        for msg in reversed(chat_gecmisi):
            if msg.type == "ai":
                son_ai_cevabi = msg.content
                break

        birlesik_soru = f"Önceki sonucun: {son_ai_cevabi}\nYeni soru: {soru}"

        # Agent'e gönder
        yanit = agent.run(birlesik_soru)
        print(f"Cevap: {yanit}")

    except Exception as e:
        print(f"Hata: {str(e)}. Lütfen geçerli bir soru giriniz.")
        continue
