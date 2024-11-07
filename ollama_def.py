from langchain_community.llms.ollama import Ollama

# host_url = os.getenv("OLLAMA_HOST")
# print(host_url)

# host_url = "host.docker.internal"
llm = Ollama(model="openhermes:latest", base_url="http://host.docker.internal:11434")
