from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from llam_app import llm


class LLMRequest:
    def __init__(self, message: str, temperature: int=0.1):
        self.message = message
        self.temperature = temperature

    def _define_prompt(self):
        template: str = """

         You are a llm that extracts useful information from summaries.  Your task is to output the information similar to named entity recognition based on the provided examples. 

        Instructions:
        - Do NOT write any code or instructions for creating a parser or chatbot.
        - ONLY provide the extracted information in JSON format.
        - Try to extract the ISO, DATE and the DISASTER as the top priority.
        - Handle dates and times accurately based on the given date of the message.

        Examples:

        Input:
        "MMR: Flood - 07-2024 - Myanmar Flood"

        Expected Output:

                ISO: MMR,
                DISASATER: Flood,
                DATE: 07-2024
            

        Input: 
        "USA: Earthquake - 2024/08/20 - San Francisco Quake"

        Expected Output:

                ISO: USA,
                DISASATER: Earthquake,
                DATE: 2024/08/20
                                

            
                

        Input: 
        "Flood in the southwest region on 5th July, 2024"

        Expected Output:

                ISO: Null,
                DISASATER: Flood,
                DATE: 5th July, 2024
                                


        Input:
        "Earthquake struck the east region of Guatemala"

        Expected Output:

                ISO: Guatemala,
                DISASATER: Earthquake,
                DATE: Null


       

        Now, extract the information from the following message:

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



