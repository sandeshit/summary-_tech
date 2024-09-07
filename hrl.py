from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
import pandas as pd
from dataf import df

# Initialize the OpenAI LLM
from llam_app import llm


# Define the prompt template to extract and convert the date
prompt = PromptTemplate(
    input_variables=["text", "today"],
    template="""

       You are a chatbot that extracts useful information from leave messages. Today's date is {today}. Your task is to extract details about late arrivals, leaves, and work-from-home (WFH) requests from the text provided and convert the dates into the format YYYY-MM-DD using today as the reference date. The reference may be given according to months, weeks, or days. Output the extracted information in JSON format.

Instructions:

    Do NOT include any code or instructions for creating a parser or chatbot.
    ONLY provide the extracted information in JSON format.
    If the name of the person is not specified, set the person as null.
    If the reason for leave is not specified, set the reason as null.
    DO NOT assume any reasons; provide null if the reason is not explicitly mentioned.
    Handle dates and times accurately based on the given reference date.
    If multiple types of leave (e.g., WFH and leave) are mentioned in a single message, output them as separate JSON objects only if they are specified for different days.
    If the person is late but does not mention a specific leave, set leave as false and provide the reason for lateness.

Examples:

Input:
XYZ: I'll be working remotely as I moved to Imadol last night. I'm not feeling well, so I'll be taking breaks to rest throughout the day..
Date: 2024-01-08

Expected Output:

json


    "name": "XYZ",
    "late": true,
    "leave": false,
    "WFH": True,
    "reason": "Not feeling well.",
    "start_date": "2024-01-08",
    "end_date": null


Input:
JOHN: I will be on leave starting tomorrow for 3 days. I have to visit the hospital.
Date: 2024-03-08

Expected Output:

json


    "name": "JOHN",
    "late": false,
    "leave": true,
    "WFH": false,
    "start_date": "2024-03-09",
    "end_date": "2024-03-12",
    "reason": "To visit hospital"


Input:
RAM: I will take a leave tomorrow as I have to visit a hospital. Also, I will be working from home the day after tomorrow.
Date: 2024-03-12

Expected Output:

json

[
    
        "name": "RAM",
        "late": false,
        "leave": true,
        "WFH": false,
        "start_date": "2024-03-13",
        "end_date": null,
        "reason": "To visit hospital"
    ,
    
        "name": "RAM",
        "late": false,
        "leave": "Second Half",
        "WFH": true,
        "start_date": "2024-03-14",
        "end_date": null,
        "reason": null
    
]

Input:
SNDSED: I will be working from home today in the first half and will be on leave on the second half. I will also be on leave for 3 days starting tomorrow.
Date: 2024-02-12

Expected Output:

json

[
    
        "name": "SNDSED",
        "late": false,
        "leave": "Second Half",
        "WFH": "First Half",
        "start_date": "2024-02-12",
        "end_date": null,
        "reason": null
    ,
    
        "name": "SNDSED",
        "late": false,
        "leave": true,
        "WFH": false,
        "start_date": "2024-02-13",
        "end_date": "2024-02-15",
        "reason": null
    
]

Now, extract the information from the following message text: {text}

    """

        
)

# Create an LLMChain using the LLM and the prompt template
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Function to get today's date
def get_today():
    return datetime.now().strftime("%Y-%m-%d")

# Define a function to parse dates using LLM
def parse_dates(text):
    today = get_today()
    return llm_chain.run({"text": text, "today": today})

# Example text with relative dates


    # Use the LLM to parse the dates
df['json_out'] = df['Message'].apply(parse_dates)

df.to_csv('/home/sandesh/Desktop/projects all/olamo/olam/output.csv')
df.to_json('/home/sandesh/Desktop/projects all/olamo/olam/output.json')




