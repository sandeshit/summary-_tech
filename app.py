from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from llam_app import llm

class LLMRequest:
    def __init__(self, message: str, temperature: int=0.1):
        self.message = message
        self.temperature = temperature

    '''def _define_prompt(self):
        template: str = """

        You should act like a chatbot that extracts the useful information from leave messages. The return should be in a json format. WFH means work from home.
        You should extract the name of the senders and the leave time and also the leave reason. The leave time should be handles correctly. If there are texts containing reference 
        times you should show the appropriate date by adding to the current date passed correctly. THE DATE IN THE INPUT IS THE DATE IN WHICH THE USER SENT THE MESSAGE. YOU SHOULD 
        PERFORM THE TIME CALCULATION WITH RESPECT TO THAT PARTICULAR DATE. Read the whole articlce correctly it may contain many inputs.
        Here are some examples for referencing how the input to you and your output should look like:

        Input:
        XYZ: Hello, Sadikshya Di, I will be late by 30 minutes today. My scooter is not starting. 
             Date: 2024-01-08

        Output of gemma:
        XYZ: 
                    late:True,
                    leave: False,
                    WFH: False,
                    reason: Scooter not starting.

        Input: 
        JOHN: I will be on leave starting tomorrow for 3 days. I have to visit hospital. 
              Date: 2024-03-08

        Output of gemma:
        JOHN: 
                late: False,
                Leave: True,
                start_date: 2024-03-09
                end_date: 2024-03-12
                reason: To visit hospital


        If multiple conditions are specified then use a list to give multiple JSON files in the list. DO NOT WRITE CODE. For example the input as the one give below.

        Input:
        SNDSED: I will be working from home today in the first half and will be on leave on the second half. I will also be on leave for 3 days starting tomorrow. 
                Date: 2024-02-12

        Output of gemma:
        [SNDSED:
                late: False,
                Leave: Second Half,
                WFH: first half,
                reason: Null,
         SNDSED:
                late: False,
                leave: True,
                start_date: 2024-02-13,
                end_date: 2024-02-15]
                


        Message: {message}

        Reply:


        DO NOT WRITE THE CODE TO MAKE A PARSER JUST ACT LIKE A EXTRACTOR ACCORDING TO THE TEMPLATE. DO NOT WRITE THE CODE TO MAKE A PARSER. IF THE REASON IS NOT SPECIFIED 
        LEAVE IT AS NULL. DO NOT GIVE A DEFAULT REASON.
        """
        return template'''
    
    def _define_prompt(self):
        template: str = """

         You are a chatbot that extracts useful information from leave messages. Your task is to output the information in JSON format based on the provided examples. 

        Instructions:
        - Do NOT write any code or instructions for creating a parser or chatbot.
        - ONLY provide the extracted information in JSON format.
        - If the reason for leave is not specified, leave it as NULL.
        - DO NOT assume any reasons. Provide NULL if the reason is not explicitly mentioned.
        - Handle dates and times accurately based on the given date of the message.

        Examples:

        Input:
        XYZ: Hello, Sadikshya Di, I will be late by 30 minutes today. My scooter is not starting. 
            Date: 2024-01-08

        Expected Output:
        XYZ: 
            
                "late": True,
                "leave": False,
                "WFH": False,
                "reason": "Scooter not starting."
            

        Input: 
        JOHN: I will be on leave starting tomorrow for 3 days. I have to visit hospital. 
            Date: 2024-03-08

        Expected Output:
        JOHN: 
            
                "late": False,
                "leave": True,
                "start_date": "2024-03-09",
                "end_date": "2024-03-12",
                "reason": "To visit hospital"

        Input: 
        RAM: I will be unavailable in the second half today. I have to visit hospital. 
            Date: 2024-03-12

        Expected Output:
        JOHN: 
            
                "late": False,
                "leave": second half,
                "start_date": "2024-03-12",
                "end_date": null,
                "reason": "To visit hospital"
            

        Input:
        SNDSED: I will be working from home today in the first half and will be on leave on the second half. I will also be on leave for 3 days starting tomorrow. 
                Date: 2024-02-12

        Expected Output:
        [
            
                "name": "SNDSED",
                "late": False,
                "leave": "Second Half",
                "WFH": "First Half",
                "reason": null
            ,
            
                "name": "SNDSED",
                "late": False,
                "leave": True,
                "start_date": "2024-02-13",
                "end_date": "2024-02-15",
                "reason": null
            
        ]

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




