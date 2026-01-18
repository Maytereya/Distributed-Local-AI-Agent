import configparser
import os

# Определяем путь к текущему файлу config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Если config.ini лежит рядом с config.py:
config_path = os.path.join(BASE_DIR, 'config.ini')

# Если config.ini лежит на уровень выше, используем '..'
# config_path = os.path.join(BASE_DIR, '..', 'config.ini')

# print("Computed config.ini path:", config_path)

config = configparser.ConfigParser()

config.read(config_path)

# print("Current working directory:", os.getcwd())
# files_read = config.read(config_path)
# print("Files read:", files_read)
# -------------------------------------------------
environment = 'DOCKER_PRODUCTION' #'PRODUCTION'
# | 'DEVELOPMENT' IP квартиры
# | 'LOCAL' localhost
# | 'PRODUCTION' внешний белый IP Клиники
# | 'DOCKER_PRODUCTION' под контейнеры

# print("environment:", environment)
# --------------------------------------------------

ollama_url = config[environment]['ollama_url']
chroma_host = config[environment]['chroma_host']
chroma_port = int(config[environment]['chroma_port'])
MEILI_URL = config[environment]['MEILI_URL']
VOSK_URL = config[environment]['VOSK_URL']
WHISPER_URL = config[environment]['WHISPER_URL']
WHISPER_HTTP_API = config[environment]['WHISPER_HTTP_API']

MASTER_KEY = config['DEFAULT']['MASTER_KEY']
AUTH_NAME = config['DEFAULT']['AUTH_NAME']
AUTH_PASS = config['DEFAULT']['AUTH_PASS']
APP_DATA_DIR = config['DEFAULT']['APP_DATA_DIR']

nayka_base_url = config['DEFAULT']['base_url']
nayka_base_url_no_site = config['DEFAULT']['base_url_no_site']
nayka_login = config['DEFAULT']['nayka_login']
nayka_pass = config['DEFAULT']['nayka_pass']
think = config['DEFAULT']['think']
giga_authorization = config['DEFAULT']['giga_authorization_key']
salut_authorization = config['DEFAULT']['salut_speech_key']
SBER_HOST = config['DEFAULT']['SBER_HOST']
SBER_CA = config['DEFAULT']['SBER_CA']
SBER_TOKEN = config['DEFAULT']['SBER_TOKEN']
SBER_MODEL = config['DEFAULT']['SBER_MODEL']
SBER_ENABLE_PARTIAL = config['DEFAULT']['SBER_ENABLE_PARTIAL']
SBER_ENABLE_MULTI_UTTERANCE = config['DEFAULT']['SBER_ENABLE_MULTI_UTTERANCE']
SBER_NO_SPEECH_TIMEOUT = int(config['DEFAULT']['SBER_NO_SPEECH_TIMEOUT'])
SBER_MAX_SPEECH_TIMEOUT = int(config['DEFAULT']['SBER_MAX_SPEECH_TIMEOUT'])
SBER_DUMP_DIR = config['DEFAULT']['SBER_DUMP_DIR']
SBER_DUMP_AUDIO = bool(config['DEFAULT']['SBER_DUMP_AUDIO'])
ll_model_big = config['DEFAULT']['ll_model_big']
ll_model_small = config['DEFAULT']['ll_model_small']