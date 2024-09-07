from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from datetime import datetime
from dataf import df

# Initialize the OpenAI LLM
from llam_app import llm

# Define prompt templates for each type of information
prompts = {
    "leave": PromptTemplate(
        input_variables=["text", "today"],
        template="""

        You are a chatbot that extracts useful information from leave messages. Today's date is {today}. Your task is to extract details from the text provided and convert the dates into the format YYYY-MM-DD using today as the reference date. The reference may be given according to months, weeks, or days. Output the extracted information in JSON format.


        **Instructions:**
        - Do not include any code, comments, or explanations.
        - Provide only the JSON data as specified below.
        - If any information is missing, set the corresponding fields to null.
        
        Extract information from the message {text} with reference to the date {today}. Provide the output in JSON format:

        
        

        Examples:

        Input:

        "Jane will be on leave from 2024-04-01 to 2024-04-05 due to personal reasons."

        Expected Output:
        
            {{"leave": [{{
                "date_from": "2024-04-01",
                "date_to": "2024-04-05",
                "reason": "Personal reasons"
            }}]}}
        

        If there is no leave information, ensure that "leave" is set to an empty list.
        """
    ),
    "first_half": PromptTemplate(
        input_variables=["text", "today"],
        template="""
        You are a chatbot that extracts useful information from leave messages. Today's date is {today}. Your task is to extract details from the text provided and convert the dates into the format YYYY-MM-DD using today as the reference date. The reference may be given according to months, weeks, or days. Output the extracted information in JSON format.


        **Instructions:**
        - Do not include any code, comments, or explanations.
        - Provide only the JSON data as specified below.
        - If any information is missing, set the corresponding fields to null.

        Extract the information related to the first half of the day from the message {text} with reference to the date {today}. Provide the output in JSON format:

        
            {{"first_half": [{{
                "date": "date",
                "reason": "reason2"
                }}]
            }}
        

        Examples:

        Input:
        "Michael will be working from home in the first half of 2024-05-12 due to a doctor's appointment."

        Expected Output:
        
            {{"first_half": [{{
                "date": "2024-05-12",
                "reason": "Doctor's appointment"
            }}]
            }}
        

        If there is no information about the first half of the day, ensure that "first_half" is set to an empty list.
        """
    ),
    "second_half": PromptTemplate(
        input_variables=["text", "today"],
        template="""

        You are a chatbot that extracts useful information from leave messages. Today's date is {today}. Your task is to extract details from the text provided and convert the dates into the format YYYY-MM-DD using today as the reference date. The reference may be given according to months, weeks, or days. Output the extracted information in JSON format.

        **Instructions:**
        - Do not include any code, comments, or explanations.
        - Provide only the JSON data as specified below.
        - If any information is missing, set the corresponding fields to null.

        Extract the information related to the second half of the day from the message {text} with reference to the date {today}. Provide the output in JSON format:

        
            {{"second_half": [{{
                "date": "date",
                "reason": "reason3"
            }}]
            }}
        

        Examples:

        Input:
        "Emily will be on leave for the second half of 2024-06-15 for a family event."

        Expected Output:
        
            {{"second_half": [{{
                "date": "2024-06-15",
                "reason": "Family event"
            }}]
            }}
        

        If there is no information about the second half of the day, ensure that "second_half" is set to an empty list.
        """
    ),
    "wfh": PromptTemplate(
        input_variables=["text", "today"],
        template="""

        You are a chatbot that extracts useful information from leave messages. Today's date is {today}. Your task is to extract details from the text provided and convert the dates into the format YYYY-MM-DD using today as the reference date. The reference may be given according to months, weeks, or days. Output the extracted information in JSON format.

        **Instructions:**
        - Do not include any code, comments, or explanations.
        - Provide only the JSON data as specified below.
        - If any information is missing, set the corresponding fields to null.

        Extract the work-from-home (WFH) information from the message {text} with reference to the date {today}. Provide the output in JSON format:

        
            {{"wfh": [{{
                "date_from": "start_date",
                "date_to": "end_date",
                "reason": "reason4"
            }}]}}
        

        Examples:

        Input:
        "Lisa will be working from home from 2024-07-01 to 2024-07-03 as she has to wait for a home repair."

        Expected Output:
        
            {{"wfh": [{{
                "date_from": "2024-07-01",
                "date_to": "2024-07-03",
                "reason": "Home repair"
            }}]}}
        

        If there is no WFH information, ensure that "wfh" is set to an empty list.
        """
    ),
    "unavailable": PromptTemplate(
        input_variables=["text", "today"],
        template="""

        You are a chatbot that extracts useful information from leave messages. Today's date is {today}. Your task is to extract details from the text provided and convert the dates into the format YYYY-MM-DD using today as the reference date. The reference may be given according to months, weeks, or days. Output the extracted information in JSON format.

        **Instructions:**
        - Do not include any code, comments, or explanations.
        - Provide only the JSON data as specified below.
        - If any information is missing, set the corresponding fields to null.
        
        Extract the information related to being unavailable (quick leave or late) from the message {text} with reference to the date {today}. Provide the output in JSON format:

        
            {{"unavailable": [{{
                "date": "date",
                "from_time": "start_time",
                "end_time": "end_time",
                "reason": "reason5"
            }}]}}
        

        Examples:

        Input:
        "John will be late to work on 2024-08-10 from 09:00 to 11:00 due to a car breakdown."

        Expected Output:
        
            {{"unavailable": [{{
                "date": "2024-08-10",
                "from_time": "09:00",
                "end_time": "11:00",
                "reason": "Car breakdown"
            }}]}}
        

        If there is no information about being unavailable, ensure that "unavailable" is set to an empty list.

        Also if there is information about multiple keys then make sure to create a list of jsons.
        """
    )
}

# Create LLMChains for each type of information
llm_chains = {key: LLMChain(llm=llm, prompt=prompt) for key, prompt in prompts.items()}

# Function to get today's date
def get_today():
    return datetime.now().strftime("%Y-%m-%d")

# Define a function to parse the text using appropriate prompt
def parse_information(text):
    today = get_today()
    output = {
        "leave": [],
        "first_half": [],
        "second_half": [],
        "wfh": [],
        "unavailable": []
    }
    for key in output.keys():
        result = llm_chains[key].run({"text": text, "today": today})
        print(result)
    

# Apply the function to your DataFrame

hello = parse_information("I will be working from home in the first half tomorrow and be on leave on for the second half")

#df['json_out'] = df['Message'].apply(parse_information)

# Save the results
df.to_csv('/home/sandesh/Desktop/projects all/olamo/olam/output_new.csv')
df.to_json('/home/sandesh/Desktop/projects all/olamo/olam/output_new.json')
