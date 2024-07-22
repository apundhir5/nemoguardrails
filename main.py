import os
import warnings
import time
import weather
from dotenv import load_dotenv
from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails
from langchain_openai import ChatOpenAI, OpenAI
# from langchain_community.llms import ChatOpenAI, set_llm_cache
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, RedisCache, RedisSemanticCache
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
import langchain
import redis

import hashlib
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
load_dotenv()

open_ai_key = os.environ.get('OPENAI_API_KEY')

# chat_model = OpenAI(model_name='gpt-4-turbo-preview', openai_api_key=open_ai_key)
chat_model = ChatOpenAI(model_name='gpt-4-turbo-preview', openai_api_key=open_ai_key)

# # In Memory Cache
# set_llm_cache(InMemoryCache())

# Redis Cache
redis_client = redis.Redis(host="localhost", port=6379, db=0)
# redis_client = redis.Redis(host="localhost", port=6379, username="default", password="password", db=0)
set_llm_cache(RedisCache(redis_client))

# Semantic Cache
# redis_url = "redis://default:password@localhost:6379" #"rediss://:" + "password" + "@"+ "localhost:6379"
redis_url = "redis://localhost:6379"
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=open_ai_key
)
set_llm_cache(RedisSemanticCache(redis_url=redis_url, embedding=embeddings, score_threshold=0.5))


# GPTCache
def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()    

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )

set_llm_cache(GPTCache(init_gptcache))


def call_rails(prompt):
    response = rails.generate(messages=[{
        "role": "user",
        "content": prompt
    }])
    print(f"RESPONSE={response['content']}")
    
    info = rails.explain()
    print(f"HISTORY=\n{info.colang_history}")
    
    # redis_dict = langchain.llm_cache._cache_dict
    # key = list(redis_dict.keys())[0]
    # print("Key:", key)
    # print("\n")
    # print(langchain.llm_cache._cache_dict[key].similarity_search_with_score(query='Hi'))
    # print("\n")
    # print(langchain.llm_cache._cache_dict[key].similarity_search_with_score(query='Hi'))
    

    # info.print_llm_calls_summary()

# config = RailsConfig.from_path("./config")
config = RailsConfig.from_path("./config-norail")

rails = LLMRails(config, llm=chat_model) #, verbose=True)
rails.register_action(action=weather.location_api, name="location_api")
rails.register_action(action=weather.weather_api, name="weather_api")

if __name__ == '__main__':
    try:
        while True:
            user_input = input("\nEnter something or Ctrl+C to stop:")
            
            start = time.process_time()
            call_rails(user_input)
            elapsed = (time.process_time() - start)
            print(f"Time taken:{elapsed}")            


    except KeyboardInterrupt:
        print("\tTerminated")
    