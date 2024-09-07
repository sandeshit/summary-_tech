from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from dataset import samsum
import pandas as pd

from scorer import rogue_score_calculate, bert_score_calculate

# Sample definition of the function

# Convert the 'test' dictionary to a DataFrame
samsum_cod_test_df = pd.DataFrame(samsum['test'])

samsum_cod_test_df = samsum_cod_test_df[:10]

# Apply the function to the 'dialogue' column to convert to Document objects

# Load the language model
llm = Ollama(model="qwen:0.5b")


def invoke(docs):

    prompt_template = """
    You will generate increasingly concise, entity-dense summaries of the given article. Repeat the following 2 steps 5 times.

        Step 1: Identify 1-3 informative entities from the article that are missing from the previously generated summary.

        Step 2: Write a new, denser summary of identical length that covers every entity and detail from the previous summary, plus the missing entities.

        Criteria for Missing Entities:

            Relevant: Related to the main story.
            Specific: Descriptive yet concise (5 words or fewer).
            Novel: Not included in the previous summary.
            Faithful: Present in the article.
            Anywhere: Located anywhere in the article.

        Guidelines:

            The first summary should be long (2-3 sentences, ~20 words) and highly non-specific, containing little information beyond the entities marked as missing. Use verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
            Make every word count: Rewrite the previous summary to improve flow and make space for additional entities.
            Make space through fusion, compression, and removal of uninformative phrases like "the article discusses."
            Summaries should become highly dense and concise yet self-contained, e.g., easily understood without the article.
            Missing entities can appear anywhere in the new summary.
            Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

        Note: Use the exact same number of words for each summary.


    {text}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Use RunnableSequence instead of LLMChain
    summary_chain = RunnableSequence(prompt | llm)

    document = Document(page_content=docs)
    docs = [document]


    result_cod = summary_chain.invoke(docs)

    return result_cod
   

# Apply `invoke` function to each row in the DataFrame
samsum_cod_test_df['summary_generated'] = samsum_cod_test_df['dialogue'].apply(invoke)

# Remove rows where summary is None or empty
samsum_cod_test_df = samsum_cod_test_df.dropna(subset=['summary'])

result_cod_rogue = rogue_score_calculate(samsum_cod_test_df['summary_generated'], samsum_cod_test_df['summary'])
result_cod_bert = bert_score_calculate(samsum_cod_test_df['summary_generated'], samsum_cod_test_df['summary'])

'''print(samsum_cod_test_df)
print(result_cod_rogue)
print(result_cod_bert)'''


print(samsum_cod_test_df['summary_generated'][0])

# Display the updated DataFram

'''# Load the summarization chain
llm = Ollama(model="qwen:0.5b")

prompt_template = """
    You will generate increasingly concise, entity-dense summaries of the given article. Repeat the following 2 steps 5 times.

        Step 1: Identify 1-3 informative entities from the article that are missing from the previously generated summary.

        Step 2: Write a new, denser summary of identical length that covers every entity and detail from the previous summary, plus the missing entities.

        Criteria for Missing Entities:

            Relevant: Related to the main story.
            Specific: Descriptive yet concise (5 words or fewer).
            Novel: Not included in the previous summary.
            Faithful: Present in the article.
            Anywhere: Located anywhere in the article.

        Guidelines:

            The first summary should be long (2-3 sentences, ~20 words) and highly non-specific, containing little information beyond the entities marked as missing. Use verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
            Make every word count: Rewrite the previous summary to improve flow and make space for additional entities.
            Make space through fusion, compression, and removal of uninformative phrases like "the article discusses."
            Summaries should become highly dense and concise yet self-contained, e.g., easily understood without the article.
            Missing entities can appear anywhere in the new summary.
            Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

        Note: Use the exact same number of words for each summary.


{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Use RunnableSequence instead of LLMChain
summary_chain = RunnableSequence(prompt | llm)

document = Document(page_content=small_paragraph)

summary = summary_chain.invoke({"text": document.page_content})'''






















