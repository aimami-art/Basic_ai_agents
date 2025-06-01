from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

user_massage = HumanMessage(
    content="Merhaba, nasilsin, yapay zeka nedir anlat."
)

responce = llm.invoke([user_massage])

print(responce.content)