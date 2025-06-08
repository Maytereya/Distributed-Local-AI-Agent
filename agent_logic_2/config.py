import configparser
import os

# Определяем путь к текущему файлу config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Если config.ini лежит рядом с config.py:
config_path = os.path.join(BASE_DIR, 'config.ini')

# Если config.ini лежит на уровень выше, используем '..'
# config_path = os.path.join(BASE_DIR, '..', 'config.ini')

print("Computed config.ini path:", config_path)

config = configparser.ConfigParser()

config.read(config_path)

print("Current working directory:", os.getcwd())
files_read = config.read(config_path)
print("Files read:", files_read)
# -------------------------------------------------
environment = 'DOCKER_PRODUCTION'
# | 'DEVELOPMENT' IP квартиры
# | 'LOCAL' localhost
# | 'PRODUCTION' внешний белый IP
# | 'DOCKER_PRODUCTION' под контейнеры

print("environment:", environment)
# --------------------------------------------------

ollama_url = config[environment]['ollama_url']
chroma_host = config[environment]['chroma_host']
chroma_port = int(config[environment]['chroma_port'])
MEILI_URL = config[environment]['MEILI_URL']

MASTER_KEY = config['DEFAULT']['MASTER_KEY']
AUTH_NAME = config['DEFAULT']['AUTH_NAME']
AUTH_PASS = config['DEFAULT']['AUTH_PASS']
nayka_base_url = config['DEFAULT']['base_url']
nayka_login = config['DEFAULT']['nayka_login']
nayka_pass = config['DEFAULT']['nayka_pass']

ll_model_big = config['DEFAULT']['ll_model_big']
ll_model_small = config['DEFAULT']['ll_model_small']
