from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
     template="{page_content}"
)
document_variable_name = "context"
llm = Ollama(model = "qwen2:latest")
# The prompt here should take as an input variable the
# `document_variable_name`
prompt = PromptTemplate.from_template(
    "Summarize this content: {context}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
# We now define how to combine these summaries
reduce_prompt = PromptTemplate.from_template(
    "Combine these summaries: {context}"
)
reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
)
chain = MapReduceDocumentsChain(
    llm_chain=llm_chain,
    reduce_documents_chain=reduce_documents_chain,
)
from langchain.chains import MapReduceDocumentsChain


# Example documents to be summarized
documents = [
    #Document(page_content="Records mention the Gopalas and Mahishapalas believed to have been the earliest rulers with their capital at Matatirtha, the south-west corner of the Kathmandu Valley. From the 7th or 8th Century B.C. the Kirantis are said to have ruled the valley. Their famous King Yalumber is even mentioned in the epic, ‘Mahabharat’. Around 300 A.D. the Lichhavis arrived from northern India and overthrew the Kirantis. One of the legacies of the Lichhavis is the Changu Narayan Temple near Bhaktapur, a UNESCO World Heritage Site (Culture), which dates back to the 5th Century. In the early 7th Century, Amshuvarma, the first Thakuri king took over the throne from his father-in-law who was a Lichhavi. He married off his daughter Bhrikuti to the famous Tibetan King Tsong Tsen Gampo thus establishing good relations with Tibet. The Lichhavis brought art and architecture to the valley but the golden age of creativity arrived in 1200 A.D with the Mallas."),
    #Document(page_content="During their 550 year rule, the Mallas built numerous temples and splendid palaces with picturesque squares. It was also during their rule that society and the cities became well organized; religious festivals were introduced and literature, music and art were encouraged. After the death of Yaksha Malla, the valley was divided into three kingdoms: Kathmandu (Kantipur), Bhaktapur (Bhadgaon) and Patan (Lalitpur). Around this time, the Nepal as we know it today was divided into about 46 independent principalities. One among these was the kingdom of Gorkha with a Shah ruler. Much of Kathmandu Valley’s history around this time was recorded by Capuchin friars who lived in the valley on their way in and out of Tibet."),
    Document(page_content="Then on 1st June 2001, a horrific tragedy wiped out the entire royal family including King Birendra and Queen Aishwarya with many of their closest relatives. With only King Birendra’s brother, Gyanendra and his family surviving, he was crowned the king. King Gyanendra abided by the elected government for some time and then dismissed the elected Parliament to wield absolute power. In April 2006, another People’s Movement was launched jointly by the democratic parties focusing most energy in Kathmandu which led to a 19-day curfew. Eventually, King Gyanendra relinquished his power and reinstated the Parliament. On November 21, 2006, Prime Minister Girija Prasad Koirala and Maoist chairman Prachanda signed the Comprehensive Peace Agreement (CPA) 2006, committing to democracy and peace for the progress of the country and people. A Constituent Assembly election was held on April 10, 2008. On May 28, 2008, the newly elected Constituent Assembly declared Nepal a Federal Democratic Republic, abolishing the 240 year-old monarchy. Nepal today has a President as Head of State and a Prime Minister heading the Government.")
]

# Assuming `chain` is already defined as in your code
# Run the chain with the documents
result = chain.invoke(documents)

# Print the result
print(result['output_text'])
