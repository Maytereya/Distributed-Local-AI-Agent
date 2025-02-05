import configparser
import os
from typing import Literal

config = configparser.ConfigParser()


def choose_exec_directory(dir: Literal["root", "sub"]):
    if dir == "sub":
        return config.read('../config.ini')

    return config.read('config.ini')


# config.read('config.ini')

print("Current working directory:", os.getcwd())
files_read = config.read('config.ini')
print("Files read:", files_read)
# print(config.sections())

environment = 'PRODUCTION'  # 'DEVELOPMENT' OR 'LOCAL'

# print("environment:", environment)

ollama_url = config[environment]['ollama_url']
chroma_host = config[environment]['chroma_host']
chroma_port = int(config[environment]['chroma_port'])

# emb_model = config['DEFAULT']['emb_model']
ll_model_big = config['DEFAULT']['ll_model_big']
# ll_model = config['DEFAULT']['ll_model']
ll_model_small = config['DEFAULT']['ll_model_small']

# ll_model_large_ctx = config['DEFAULT']['ll_model_large_ctx']
# ll_model_large_ctx_70b = config['DEFAULT']['ll_model_large_ctx_70b']

# ll_model_llama31_70b_instruct_q8 = config['DEFAULT']['ll_model_llama31_70b_instruct_q8']
# question1 = config['DEFAULT']['question1']
# collect_name = config['DEFAULT']['collect_name']
