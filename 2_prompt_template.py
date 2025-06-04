from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
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

konu = "yapay zeka nedir"

prompt = PromptTemplate(
    input_variables=["user_message"],
    template="Kısa ve anlaşılır bir şekilde {user_message} açıkla"
)

formatted_prompt = prompt.format(user_message=konu)

response = llm.invoke([HumanMessage(content=formatted_prompt)])

print(response.content)