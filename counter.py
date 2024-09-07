from langchain_core.prompts import PromptTemplate

from langchain.chains import LLMChain
from llam_app import llm


class LLMRequest:
    def __init__(self, message: str, temperature: int=0.1):
        self.message = message
        self.temperature = temperature

    def _define_prompt(self):
        template: str = """
        You are an LLM that checks whether the provided summary contains all the necessary information specified in the given fields.
Instructions:

    Do NOT WRITE ANY CODE FOR CREATING A PROGRAM OR FUNCTION TO DO THE JOB. JUST ACT LIKE A INFORMATION EXTRACTOR.
    ONLY provide the output in JSON format, indicating whether the required information is present or missing.
    Prioritize checking the presence of iso3,date, and name (from dtype_data).
    If any of the three fields is missing then output that the summary does not contain the require information. 
    Handle dates and times accurately based on the provided message data.

Check the Following:

    iso3: Get the iso3 name from the iso3 field and check if it is present in the summary field or not. iso3 is generally a three lettered all capital country code. It may be present anywhere in the summary. 
    date: Confirm the presence of the date in the summary. The date can be in any format, for example: 2022/12/10, 2023-09-12, 2022-09, 9th July, 2024.
    name (from dtype_data): Confirm the disaster type in the summary from the name field of the dtype_data.

    iso3: COVID-19
    iso3: COVID-19 #Field-Report-Number (Date)
    iso3: Disaster - Date SUMMARY #Field Report Number (Date)
    iso3: Disaster - Date SUMMARY

Flow of Program:

    First, check if all the fields are present or not. All of the fields should contain values other than null.
    Then, check the summary to see if it matches any of the 4 summary formats.
    We are only concerned if it does not match any formats.If it does not match the format do the next step.
    Check if the prioritized fields (iso3, name from dtype_data, any type of date) are present or not in the summary.

  Here is the list of some iso3 : 
    "AFG", "ALB", "DZA", "AND", "AGO", "ARG", "ARM", "AUS", "AUT", "AZE",
    "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BTN", "BOL",
    "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI", "CPV", "KHM", "CMR",
    "CAN", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COG", "COD", "CRI",
    "CIV", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI", "DMA", "DOM", "ECU",
    "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI", "FIN", "FRA",
    "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN", "GNB",
    "GUY", "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL",
    "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "PRK", "KOR",
    "KWT", "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU",
    "LUX", "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS",
    "MEX", "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM",
    "NRU", "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "MKD", "NOR", "OMN",
    "PAK", "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT",
    "ROU", "RUS", "RWA", "KNA", "LCA", "VCT", "WSM", "SMR", "STP", "SAU",
    "SEN", "SRB", "SYC", "SLE", "SGP", "SVK", "SVN", "SLB", "SOM", "ZAF",
    "SSD", "ESP", "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", "TWN", "TJK",
    "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV",
    "UGA", "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VEN", "VNM",
    "YEM", "ZMB", "ZWE","GTM", "PSE"

Here are some examples:

Input Example:

json


    "countries_data": [
        
            "id": 220,
            "iso3": "NEP",
            "name": "Nepal"
        
    ],
    "dtype_data": 
        "id": 1,
        "name": "Nepal Earthquake"
    ,
    "event_data": 
        "id": 5845,
        "name": "Palestine Cold wave and flooding -1/2022",
    "f_start_date": "2022-01-26 12:01:00",
    "f_report_date": "2022-02-21 08:02:14",
    "summary": "Earthquake in the easter region of Nepal NEP - 1/2022"



Expected Output Example:

json


    "status": "Complete",
    "message": "Structure mismatch but the summary contans all the information."


Input Example:

json


    "countries_data": [
        
            "id": 205,
            "iso3": "PSE",
            "name": "Palestine"
        
    ],
    "dtype_data": 
        "id": 1,
        "name": "Cold Wave"
    ,
    "event_data": 
        "id": 5845,
        "name": "Palestine Cold wave and flooding -1/2022",
    "f_start_date": "2022-01-26 12:01:00",
    "f_report_date": "2022-02-21 08:02:14",
    "summary": "PSE Cold wave and flooding - 1/2022"



Expected Output Example:

json


    "status": "Incomplete",
    "message": "Structure mismatch but the summary contains all the information."


Another Input Example:

json


    "countries_data": [
        
            "id": 136,
            "iso3": "PHL",
            "name": "Philippines"
        
    ],
    "dtype_data": 
        "id": 4,
        "name": "Cyclone"
    ,
    "event_data": 
        "id": null,
        "name": null,
    "f_start_date": "2022-12-10 7:00:23",
    "f_report_date": "2022-12-10 12:09:12",
    "summary": "2022/12/10 Cyclone - Cyclone hit the southern part of Philippines"



Expected Output Example:

json


    "status": "Incomplete",
    "message": "there is no iso3 and no presence of any date"


Another Input Example:

json


    "countries_data": [
        
            "id": 108,
            "iso3": "USA",
            "name": "United States of America"
        
    ],
    "dtype_data": 
        "id": 4,
        "name": "Cyclone"
    ,
    "event_data": 
        "id": null,
        "name": null,
    "f_start_date": "2022-12-10 7:00:23",
    "f_report_date": "2022-12-10 12:09:12",
    "summary": "2022/12/10 northern part of Americas"



Expected Output Example:

json


    "status": "Incomplete",
    "message": "there is no iso3 and no presence of any disaster also"


Another Input Example:

json


    "countries_data": [
        
            "id": 205,
            "iso3": "PSE",
            "name": "Palestine"
        
    ],
    "dtype_data": 
        "id": 1,
        "name": "Cold Wave"
    ,
    "event_data": 
        "id": 5845,
        "name": "Palestine Cold wave and flooding -1/2022",
    "f_start_date": "2022-01-26 12:01:00",
    "f_report_date": "2022-02-21 08:02:14",
    "summary": "PSE: Cold Wave: Cold Wave hit the northern region of Palestine. 2022-02"
    


Expected Output Example:

json


    "status": "Incomplete",
    "message": "Structure mismatch but all the informations is present"



 
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
        
user_message = """
    
    {
    "countries_data": [
        {
            "id": 76,
            "iso3": "GTM",
            "name": "Guatemala"
        }
    ],
    "dtype_data": {
        "id": 2,
        "name": "Earthquake"
    },
    "event_data": {
        "id": 5836,
        "name": "Guatemala: Sismos sensibles",
    "f_start_date": "2022-02-16 12:02:00",
    "f_report_date": "2022-02-16 08:02:12",
    "summary": "the southern part Guatemala"
    }
}
 """ 


llmreqeuest = LLMRequest(message= user_message)
response = llmreqeuest()
print(response['text'])



