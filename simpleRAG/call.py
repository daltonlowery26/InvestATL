# %% load transformer 
from matplotlib.style import context
import polars as pl
import os
import requests
import dotenv
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from sklearn.neighbors import NearestNeighbors
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Emory/InvestATL/data')
dotenv.load_dotenv()

# %% load model and create pipeline
checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to('cuda')

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.1, # prevent randomness
    top_p=0.95,
    repetition_penalty=1.15
)

local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# %% prepare johns creek data
cen = pl.read_csv('johnCreekCen.csv')
pond = pl.read_csv('LakesPonds.csv')
cen = cen.select(pl.all().name.prefix('census_'))
pond = pond.select(pl.all().name.prefix('pond_lakes_'))
cen_pond = pl.concat([cen, pond], how="horizontal")

# %% prepare cencus cols
with open('variables.json', 'r') as f:
    census = json.load(f) 
extracted_items = [
    (k, v["concept"]) 
    for k, v in census.get("variables", {}).items() 
    if "concept" in v
]
# keys and labels list
keys_list, labels_list = zip(*extracted_items)

# %% census  embeddings 
search_embeddings = embedding_model.encode(labels_list)

# %%  cols embeddings 
cols = cen_pond.columns
col_embeddings = embedding_model.encode(cols)

# %% output loop
def get_answer(user_question):
    # question embed
    question_embed = embedding_model.encode(user_question)
    
    # col simil
    scores = np.dot(col_embeddings, question_embed.T) # dot product similairty
    best_idx = np.argmax(scores)
    matched_column = cols[best_idx]
    
    # search simialirty
    search_scores = np.dot(search_embeddings, question_embed.T) # dot product similairty
    s_best_idx = np.argmax(search_scores)
    matched_search = labels_list[s_best_idx]
    matched_key = keys_list[s_best_idx]
    
    print(f"Matched Question '{user_question}' to Column '{matched_column}'")
    print(f"Matched Question '{user_question}' to Search '{matched_search}'")
    
    # retrive data from local csvs
    context_data = cen_pond.select(pl.col(matched_column)).drop_nulls().to_series()
    context_data = json.dumps(context_data.to_list())
    context_data = matched_column + ": " + context_data
    print(context_data)
    
    # cencus data 
    url = f"https://api.census.gov/data/2020/dec/sdhc?get={matched_key}&for=state:13&key={os.getenv('CENSUS_API')}"
    
    # get request
    response = requests.get(url)
    
    # check connection and parse request
    if response.status_code == 200:
        census_data = response.json()
    else:
        print(f"Request failed: {response.status_code} - {response.text}")
    
    census_data = json.dumps(census_data)
    print(census_data)
    census_data = matched_search + ": " + census_data
    # generate data
    template = """
    ### Instructions
        1. **Source Material Only**: Use ONLY the provided "Data" and "Census" text to answer. Do not use outside knowledge.
        2. **"I dont know" Rule**: If the answer is not explicitly in the Data or Census, or if the data is missing, output exactly "idk".
        3. **No Interpretation**: Do not infer information. If the specific value is not written, it does not exist.
        4. **Calculations**: Do not perform calculations unless the question starts with "How many". In that case, count the valid occurrences.
        
        ### Examples
        
        Data: [AMOUNT OF DOGS: 1]
        Census: []
        Question: How many dogs are there?
        Answer: 1
        
        Data: [HOUSE ON LAND: True, True, False, True, Valid]
        Census: []
        Question: How many houses?
        Answer: 4
        
        Data: [DOGS: 3 6]
        Census: [GA_POP: 10m]
        Question: How many dogs in GA?
        Answer: 3 or 6
        
        Data: [Inventory: Apples, Oranges]
        Census: []
        Question: What is the price of Bananas?
        Answer: idk
        
        ### Current Task
        Data: {context_data}
        Census: {us_census}
        Question: {question}
        Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context_data", "us_census", "question"])
    chain = prompt | local_llm 
    
    response = chain.invoke({"context_data": context_data, "us_census":census_data ,"question": user_question})
    return response

# %% questions answer
response = get_answer("how many single family homes")
print(response)
