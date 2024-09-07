from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager

llm = Ollama(model = "openhermes:latest", base_url= "http://localhost:11434")