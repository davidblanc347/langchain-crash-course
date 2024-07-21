# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="llama3-latest") # llama3-latest
prompt = ChatPromptTemplate.from_template("Tell me five pertinent facts about  {topic}")
chain = prompt | llm | StrOutputParser()
result=chain.invoke({"topic": "charles sanders peirce"})

print("\n\nFull result:")
print(result)
print("Content only:")
print(result.content)
print(result.response_metadata)
