import requests
import json
from typing import Dict, List, Optional
import logging
import sys
import os
from datetime import datetime, timedelta
import glob

# Добавляем родительскую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_logic_2 import config as c
from agent_logic_2.nayka_api.api_nayka import get_today_str

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы для кэширования
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'apidata')
CACHE_EXPIRY = timedelta(hours=24)  # Кэш действителен 24 часа


def get_cache_filename(date: str = None) -> str:
    """Возвращает имя файла кэша с датой"""
    if date is None:
        date = get_today_str()
    return os.path.join(CACHE_DIR, f'zametka_button_{date}.json')


def find_latest_cache_file() -> Optional[str]:
    """Находит самый свежий файл кэша заметок"""
    pattern = os.path.join(CACHE_DIR, 'zametka_button_*.json')
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Сортируем по дате в имени файла (последний элемент после _)
    files.sort(key=lambda x: x.split('_')[-1].replace('.json', ''), reverse=True)
    return files[0] if files else None


def cleanup_old_cache_files():
    """Удаляет все старые файлы кэша заметок, кроме сегодняшнего"""
    today = get_today_str()
    pattern = os.path.join(CACHE_DIR, 'zametka_button_*.json')
    
    for file_path in glob.glob(pattern):
        try:
            # Извлекаем дату из имени файла
            filename = os.path.basename(file_path)
            file_date = filename.split('_')[-1].replace('.json', '')
            
            if file_date != today:
                os.remove(file_path)
                logger.info(f"🗑️ Удален старый файл кэша: {filename}")
        except Exception as e:
            logger.error(f"Ошибка при удалении файла {file_path}: {e}")


def load_from_cache() -> Optional[List[Dict]]:
    """
    Загружает данные из кэша, если он существует и не устарел.
    
    Returns:
        Optional[List[Dict]]: Данные из кэша или None, если кэш недействителен
    """
    try:
        # Ищем самый свежий файл кэша
        cache_file = find_latest_cache_file()
        if not cache_file:
            return None
            
        # Проверяем время последнего обновления файла
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time > CACHE_EXPIRY:
            logger.info("Кэш устарел, требуется обновление")
            return None
            
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Загружены данные о {len(data)} врачах из кэша: {os.path.basename(cache_file)}")
            return data
            
    except Exception as e:
        logger.error(f"Ошибка при чтении кэша: {e}")
        return None


def save_to_cache(data: List[Dict]) -> None:
    """
    Сохраняет данные в кэш с датой в имени файла.
    
    Args:
        data (List[Dict]): Данные для сохранения
    """
    try:
        # Создаем директорию, если её нет
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Очищаем старые файлы перед сохранением нового
        cleanup_old_cache_files()
        
        # Создаем файл с датой в имени
        cache_file = get_cache_filename()
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Данные о {len(data)} врачах сохранены в кэш: {os.path.basename(cache_file)}")
            
    except Exception as e:
        logger.error(f"Ошибка при сохранении кэша: {e}")


def get_doctors_cc_info() -> List[Dict]:
    """
    Получает информацию о заметках call-центра по врачам.
    Сначала проверяет кэш, если данных нет или они устарели - запрашивает API.
    
    Returns:
        List[Dict]: Список словарей с информацией о врачах и их заметках
        Каждый словарь содержит:
        - id: код врача
        - callCenterInfo: заметка call-центра по этому врачу
    """
    # Пробуем загрузить из кэша
    cached_data = load_from_cache()
    if cached_data is not None:
        return cached_data
        
    try:
        # URL для API
        url = "https://tc.naykalab.ru:444/H8PdIkzEjteo5ZPvVwt29t4TVjf0XN1K/medserver-test/api/v1/ai/doctors-cc-info"
        
        # Данные для авторизации
        auth = (c.nayka_login, c.nayka_pass)
        
        # Отправляем GET запрос с авторизацией
        response = requests.get(url, auth=auth, verify=False)
        
        # Проверяем статус ответа
        response.raise_for_status()
        
        # Парсим JSON ответ
        data = response.json()
        
        # Сохраняем в кэш
        save_to_cache(data)
        
        # Логируем успешное получение данных
        logger.info(f"Успешно получены данные о {len(data)} врачах")
        
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при парсинге JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return []


def get_doctor_cc_info_by_id(doctor_id: int) -> Optional[Dict]:
    """
    Получает информацию о заметках call-центра для конкретного врача по его ID.
    
    Args:
        doctor_id (int): ID врача
        
    Returns:
        Optional[Dict]: Словарь с информацией о враче и его заметках или None,
        если врач не найден
    """
    try:
        # Получаем все данные
        all_doctors = get_doctors_cc_info()
        
        # Ищем врача по ID
        for doctor in all_doctors:
            if doctor.get('id') == doctor_id:
                return doctor
                
        logger.warning(f"Врач с ID {doctor_id} не найден")
        return None
        
    except Exception as e:
        logger.error(f"Ошибка при поиске врача по ID {doctor_id}: {e}")
        return None


if __name__ == "__main__":
    # Отключаем предупреждения о небезопасном SSL
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Пример использования
    doctors_info = get_doctors_cc_info()
    print(f"Получена информация о {len(doctors_info)} врачах")
    
    # Пример получения информации о конкретном враче
    if doctors_info:
        first_doctor_id = doctors_info[0]['id']
        doctor_info = get_doctor_cc_info_by_id(first_doctor_id)
        if doctor_info:
            print(f"\nИнформация о враче с ID {first_doctor_id}:")
            print(f"Заметка call-центра: "
                  f"{doctor_info.get('callCenterInfo', 'Нет заметок')}")
 