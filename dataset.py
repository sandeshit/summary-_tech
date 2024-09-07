from datasets import load_dataset
from transformers import pipeline
import pandas as pd


samsum = load_dataset('samsum', trust_remote_code= True)

samsum_test_df = pd.DataFrame(samsum['test'])


