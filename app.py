from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from llam_app import llm

class LLMRequest:
    def __init__(self, message: str, temperature: int=0.1):
        self.message = message
        self.temperature = temperature

    def _define_prompt(self):
        template: str = """
        You act as a sentence classifier. Classify the sentiment of the sentence as postive, negative or neutral.

        Message: {message}

        Reply:
        """
        return template

    def __call__(self):
        template = self._define_prompt()
        prompt = PromptTemplate.from_template(template)
        #chain = prompt | llm | StrOutputParser
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            
        )

        response = chain.invoke({"message": self.message})

        return response
    
user_message = input("Enter the text of the user ")


llmreqeuest = LLMRequest(message= user_message)
response = llmreqeuest()
print(response['text'])




