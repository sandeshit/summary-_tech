from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from dataset import samsum
from scorer import rogue_score_calculate, bert_score_calculate
from langchain.chains import MapReduceDocumentsChain
# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.

import pandas as pd # Adjust these imports to your actual modules


# Sample definition of the function

# Convert the 'test' dictionary to a DataFrame
samsum_map_test_df = pd.DataFrame(samsum['test'])

samsum_map_test_df = samsum_map_test_df[:10]


def invoke(document):
    paragraph_docs = Document(page_content= document)

    docs = [paragraph_docs]
    
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    document_variable_name = "context"
    llm = Ollama(model = "qwen:0.5b")
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
 

    # Assuming `chain` is already defined as in your code
    # Run the chain with the documents
    result = chain.invoke(docs)


    return result['output_text']

   

# Apply `invoke` function to each row in the DataFrame
samsum_map_test_df['summary_generated'] = samsum_map_test_df['dialogue'].apply(invoke)

# Remove rows where summary is None or empty
samsum_map_test_df = samsum_map_test_df.dropna(subset=['summary'])

rogue_result_map = rogue_score_calculate(samsum_map_test_df['summary_generated'], samsum_map_test_df['summary'])
bert_result_map = bert_score_calculate(samsum_map_test_df['summary_generated'], samsum_map_test_df['summary'])

print(rogue_result_map)
print(bert_result_map)