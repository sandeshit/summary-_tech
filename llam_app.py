from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager

llm = Ollama(model = "qwen2:latest", base_url= "http://localhost:11434")