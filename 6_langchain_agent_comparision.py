"""
langchain agent tip karsilastirma

2 tool kullanimi
1. zero-shot-react-description
2. openai-functions
"""

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=openai_api_key
)

@tool
def toplama_araci(input:str) -> str:
    """
    Toplama aracini kullanarak iki sayiyi toplayin.
    """
    try:
        a,b = [int(x.strip()) for x in input.split("ve")]
        return f"Toplam {a} ve {b} = {a + b}"
    except Exception as e:
        return f"Toplama aracinda hata: {e}"
    
tools = [toplama_araci]

#zero-shot-react-description
agent_zero_shot = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

#openai-functions
agent_openai_functions = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

soru = "5 ve 8 sayilarini topla" 

print("Zero-shot agent cevabı:")
yanit_zero_shot = agent_zero_shot.run(soru)

print("OpenAI Functions agent cevabı:")
yanit_functions = agent_openai_functions.run(soru) 
