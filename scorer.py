from evaluate import load


# Assuming samsum_test_df is a DataFrame with 'summary_generated' and 'summary' columns
# Create a sample DataFrame for demonstration
# Load the ROUGE metric
rouge = load('rouge')
bertscore = load('bertscore')

# Compute ROUGE scores
def rogue_score_calculate(df1, df2):
    rogue_score = rouge.compute(predictions= df1.tolist(), references=df2.tolist())
    return rogue_score

def bert_score_calculate(df1,df2):
    bert_score = bertscore.compute(predictions= df1.tolist(), references= df2.tolist() , lang = "en")
    return bert_score



