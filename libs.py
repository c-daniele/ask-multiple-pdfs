import time
import pandas as pd
import os
from langchain_community.llms import Bedrock
from urllib.parse import quote_plus
import boto3
from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


#FIXME: remove hardcoding
def create_llm(model):

    model_kwargs = None

    if(model["provider"] == "OpenAI"):
        return ChatOpenAI(model_name=model["model_id"])

    if model["model_id"] in ('anthropic.claude-v1','anthropic.claude-v2'):
        model_kwargs = {"temperature": 0.0
            , "max_tokens_to_sample": 1000
            , "top_p": 0.1
            #, "top_k": 1
        }
    elif model["model_id"] in ('amazon.titan-tg1-large', 'amazon.titan-text-express-v1'):
        model_kwargs = { 
           "maxTokenCount": 1024, 
           "stopSequences": [], 
           "temperature": 0, 
           "topP": 0.9 
       }
    elif model["model_id"] in ('ai21.j2-ultra-v1', 'ai21.j2-ultra'):
        model_kwargs = None
    
    llm = Bedrock(
        credentials_profile_name=os.environ['AWS_PROFILE']
        , region_name=os.environ['AWS_REGION']
        , model_id=model["model_id"]
        , model_kwargs=model_kwargs)
    
    return llm


def create_embedding_model(model):

    if(model["provider"] == "OpenAI"):
        return OpenAIEmbeddings(model=model["model_id"])

    return BedrockEmbeddings(
        credentials_profile_name=os.environ['AWS_PROFILE']
        , region_name=os.environ['AWS_REGION']
        , model_id=model["model_id"]
    )


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
def get_available_LLMs():

    retVal=[]
    if("OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] != None):
        retVal.append({"model_id": "gpt-4-turbo-preview", "label": "GPT-4-turbo-preview", "provider": "OpenAI"})
        retVal.append({"model_id": "gpt-4", "label": "GPT-4", "provider": "OpenAI"})
        retVal.append({"model_id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "provider": "OpenAI"})
        retVal.append({"model_id": "babbage-002", "label": "Babbage-002", "provider": "OpenAI"})
        retVal.append({"model_id": "davinci-002", "label": "DaVinci-002", "provider": "OpenAI"})


    session = boto3.Session(profile_name=os.environ['AWS_PROFILE'], region_name=os.environ['AWS_REGION'])

    bedrock = session.client(
        service_name='bedrock'
        #, region_name=os.environ['AWS_REGION']
    )

    fm = bedrock.list_foundation_models()
    for i in fm['modelSummaries']:

        # discard embedding and dummy model returned from the Bedrock API
        if(
            (i['modelId'].find(":") > 0)
            or (i['modelId'].find("embed-") > 0)
            or (i['modelId'].find("stable-diffusion") > 0)
            or i['modelId'] in ['meta.llama2-13b-v1', 'meta.llama2-70b-v1', 'amazon.titan-image-generator-v1']
        ):
            continue

        label=i['modelName']
        if (i['modelId'] == 'anthropic.claude-v1'):
            label = 'Claude V1'
        elif (i['modelId'] == 'anthropic.claude-v2'):
            label = 'Claude V2'
        retVal.append({"model_id": f"{i['modelId']}", "label": label, "provider": "AWS"})
    
    return retVal

def get_embedding_models():

    retVal=[]
    if("OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] != None):
        retVal.append({"model_id": "text-embedding-3-large", "label": "OpenAI - Embedding V3 large", "provider": "OpenAI"})
        retVal.append({"model_id": "text-embedding-3-small", "label": "OpenAI - Embedding V3 small", "provider": "OpenAI"})
        retVal.append({"model_id": "text-embedding-ada-002", "label": "OpenAI - 2nd generation embedding", "provider": "OpenAI"})

    session = boto3.Session(profile_name=os.environ['AWS_PROFILE'], region_name=os.environ['AWS_REGION'])

    bedrock = session.client(
        service_name='bedrock'
        #, region_name=os.environ['AWS_REGION']
    )

    fm = bedrock.list_foundation_models()
    for i in fm['modelSummaries']:

        # discard dummy model returned from the Bedrock API
        if(
            (i['modelId'].find(":") > 0)
            or (i['modelId'].find("embed-") == -1)
            or (i['modelId'].find("stable-diffusion") > 0)
            or i['modelId'] in ['meta.llama2-13b-v1', 'meta.llama2-70b-v1', 'amazon.titan-image-generator-v1']
        ):
            continue
        
        retVal.append({"model_id": f"{i['modelId']}", "label": f"{i['modelName']}", "provider": "AWS"})
    
    return retVal
