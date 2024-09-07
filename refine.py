from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from dataset import samsum

import pandas as pd # Adjust these imports to your actual modules
from scorer import rogue_score_calculate, bert_score_calculate

# Sample definition of the function

# Convert the 'test' dictionary to a DataFrame
samsum_refine_test_df = pd.DataFrame(samsum['test'])

samsum_refine_test_df = samsum_refine_test_df[:10]

# Apply the function to the 'dialogue' column to convert to Document objects

# Load the language model
llm = Ollama(model="qwen:0.5b")


def invoke(document):
    paragraph_docs = Document(page_content= document)

    docs = [paragraph_docs]
    chain_refine = load_summarize_chain(llm, chain_type = "refine")

    result_refine = chain_refine.invoke(docs)

    return result_refine['output_text']
   

# Apply `invoke` function to each row in the DataFrame
samsum_refine_test_df['summary_generated'] = samsum_refine_test_df['dialogue'].apply(invoke)

# Remove rows where summary is None or empty
samsum_refine_test_df = samsum_refine_test_df.dropna(subset=['summary'])

result_refine_rogue = rogue_score_calculate(samsum_refine_test_df['summary_generated'], samsum_refine_test_df['summary'])
result_refine_bert = bert_score_calculate(samsum_refine_test_df['summary_generated'], samsum_refine_test_df['summary'])

print(result_refine_rogue)
print(result_refine_bert)

# Display the updated DataFrame


