import time
import pandas as pd
import os
from langchain_community.llms import Bedrock
from urllib.parse import quote_plus
import boto3


#FIXME: remove hardcoding
def create_llm(model_name):

    model_kwargs = None

    if model_name in ('anthropic.claude-v1','anthropic.claude-v2'):
        model_kwargs = {"temperature": 0.0
            , "max_tokens_to_sample": 1000
            , "top_p": 0.1
            #, "top_k": 1
        }
    elif model_name in ('amazon.titan-tg1-large', 'amazon.titan-text-express-v1'):
        model_kwargs = { 
           "maxTokenCount": 1024, 
           "stopSequences": [], 
           "temperature": 0, 
           "topP": 0.9 
       }
    elif model_name in ('ai21.j2-ultra-v1', 'ai21.j2-ultra'):
        model_kwargs = None
    
    llm = Bedrock(
        credentials_profile_name=os.environ['AWS_PROFILE']
        #region_name="us-west-2", 
        #endpoint_url="https://bedrock.us-west-2.amazonaws.com", 
        , region_name=os.environ['AWS_REGION']
        #, endpoint_url="bedrock.us-west-2.amazonaws.com"
        , model_id=model_name
        , model_kwargs=model_kwargs)
    
    return llm


"""
AVAILABLE MODELS
amazon.titan-tg1-large - Titan Text Large
amazon.titan-e1t-medium - Titan Text Embeddings
amazon.titan-embed-g1-text-02 - Titan Text Embeddings v2
amazon.titan-text-lite-v1 - Titan Text G1 - Lite
amazon.titan-text-express-v1 - Titan Text G1 - Express
amazon.titan-embed-text-v1 - Titan Embeddings G1 - Text
stability.stable-diffusion-xl - Stable Diffusion XL
stability.stable-diffusion-xl-v0 - Stable Diffusion XL
ai21.j2-grande-instruct - J2 Grande Instruct
ai21.j2-jumbo-instruct - J2 Jumbo Instruct
ai21.j2-mid - Jurassic-2 Mid
ai21.j2-mid-v1 - Jurassic-2 Mid
ai21.j2-ultra - Jurassic-2 Ultra
ai21.j2-ultra-v1 - Jurassic-2 Ultra
anthropic.claude-instant-v1 - Claude Instant
anthropic.claude-v1 - Claude
anthropic.claude-v2 - Claude
cohere.command-text-v14 - Command
"""
def getAvailableLLMs():
    session = boto3.Session(profile_name=os.environ['AWS_PROFILE'], region_name=os.environ['AWS_REGION'])

    bedrock = session.client(
        service_name='bedrock'
        #, region_name=os.environ['AWS_REGION']
    )

    fm = bedrock.list_foundation_models()
    retVal=[]
    for i in fm['modelSummaries']:

        label=i['modelName']
        if (i['modelId'] == 'anthropic.claude-v1'):
            label = 'Claude V1'
        elif (i['modelId'] == 'anthropic.claude-v2'):
            label = 'Claude V2'
        retVal.append({"value": f"{i['modelId']}", "label": label})
    
    return retVal
