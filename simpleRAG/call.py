# %% load transformer 
import polars as pl
import os
import requests
import dotenv
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Emory/InvestATL/data')
dotenv.load_dotenv()

# %% load model and create pipeline
checkpoint = model_name = "Qwen/Qwen3-8B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # nfp4 to reduce memory footprint
    bnb_4bit_compute_dtype=torch.float16,  # compte in fp16
    bnb_4bit_use_double_quant=True,
)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# load model, map to gpu
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto" 
)

# %% text gen pipeline
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
    context_data = matched_column + " in johns creek ATL GA" + ": " + context_data
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
    census_data = matched_search + "in GA : " + census_data
    # generate data
    template = """
    ### Instructions
        1. **Source Material Only**: Use ONLY the provided "Data" and "Census" text to answer. Do not use outside knowledge.
        2. **"I dont know" Rule**: If the answer is not explicitly in the Data or Census, or if the data is missing, output exactly "idk".
        3. **No Interpretation**: Do not infer information. If the specific value is not written, it does not exist.
        4. "Keep answers concise as possible"
        
        ### Example 1:
        Data: Johsn creek Population: [3030]
        Census: GA_population_inCity: [303002]
        Question: How many people in johns creek
        Answer: 3030
        
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
response = get_answer("population of GA")
print(response)
