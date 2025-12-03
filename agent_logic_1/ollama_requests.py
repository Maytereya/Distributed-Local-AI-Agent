# Ollama tool call with requests v 0.1
# ------------------------------------
import requests
import urllib3
from requests.auth import HTTPBasicAuth

auth = HTTPBasicAuth("ai", "aWgl7afvs5nAj00yATqUYFLDsSvemJdr")
base_url = "https://tc.naykalab.ru:444/H8PdIkzEjteo5ZPvVwt29t4TVjf0XN1K/medserver-test/api/v1/site/"


# url1 = "https://tc.naykalab.ru:444/H8PdIkzEjteo5ZPvVwt29t4TVjf0XN1K/medserver-test/api/v1/site/doctors"
# url2 = "https://tc.naykalab.ru:444/H8PdIkzEjteo5ZPvVwt29t4TVjf0XN1K/medserver-test/api/v1/site/companyUnits"


# ----------------------
#
# ----------------------
def get_all_units():
    """
    Назначение: вернуть список направлений и отделений (например, урология, кардиология и т.п.)
	Endpoint: /companyUnits
	LLM usage: подсказать пользователю, какие направления доступны или
	использовать для последующей фильтрации врачей
    :return: json

    """
    urllib3.disable_warnings()
    url = f"{base_url}/companyUnits"
    response = requests.get(url, auth=auth, verify=False)
    # response.raise_for_status()
    return response.json()


# ----------------------
#
# ----------------------
def get_doctors_by_unit(unit_name: str):
    urllib3.disable_warnings()
    """
    Назначение: вернуть всех врачей, работающих в выбранном направлении

	1.	Получить id из companyUnits по unit_name
	2.	Отфильтровать doctorCompanyUnits по companyUnit
	3.	По worker найти карточку врача в /doctors

    :param unit_name:
    :return:
    """
    # 1. Получаем список направлений
    units = get_all_units()
    unit = next((u for u in units if u['name'].lower() == unit_name.lower()), None)
    if not unit:
        return {"Ошибка": f"Направление '{unit_name}' не найдено."}

    unit_id = unit['id']

    # 2. Получаем doctorCompanyUnits
    url = f"{base_url}/doctorCompanyUnits"
    response = requests.get(url, auth=auth, verify=False)
    # response.raise_for_status()
    mappings = response.json()

    # 3. Фильтруем по companyUnit
    matching_doctors = [m for m in mappings if m['companyUnit'] == unit_id]
    worker_ids = [d['worker'] for d in matching_doctors]

    # 4. Получаем всех врачей
    url = f"{base_url}/doctors"
    response = requests.get(url, auth=auth, verify=False)
    response.raise_for_status()
    all_doctors = response.json()

    # 5. Возвращаем врачей из нужного подразделения
    filtered_doctors = [doc for doc in all_doctors if doc['id'] in worker_ids]

    return filtered_doctors


def get_service_info(service_id: int):
    """
    Назначение: показать, какие услуги врач оказывает (если можно связать врача с ID услуги)
	Вопрос: у тебя есть способ получить service_id по doctor_id?
    → Если нет, тогда get_service_info используется отдельно (например: “Расскажи про услугу ‘МРТ головы’”)

    :param service_id:
    :return:
    """
    url = f"{base_url}/api/v1/site/serviceInfo/{service_id}"
    response = requests.get(url, auth=auth, verify=False)
    if response.status_code == 404:
        return {"error": f"Услуга с ID {service_id} не найдена."}
    response.raise_for_status()
    return response.json()


def call_ollama_with_functions(question: str):
    url = "http://192.168.1.42:11434/api/chat"

    payload = {
        "model": "llama3.3:70b-instruct-q8_0",
        "messages": [
            {"role": "user", "content": question}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_all_units",
                    "description": "Вернуть список всех направлений медицины и отделений клиники",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doctors_by_unit",
                    "description": "Получить всех врачей, работающих в указанном направлении клиники",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unit_name": {
                                "type": "string",
                                "description": "Название направления медицины (например, Урология)"
                            }
                        },
                        "required": ["unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_service_info",
                    "description": "Получить информацию об услуге или анализе по ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_id": {
                                "type": "integer",
                                "description": "ID услуги или анализа"
                            }
                        },
                        "required": ["service_id"]
                    }
                }
            }
        ],
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def extract_tool_call_info(result: dict):
    """
    Выделяет имя функции и аргумент, который надо ей передать.

    :param result: Словарь, который возвращает LLM в ответ на запрос.
    :return: Имя, аргумент.
    """
    tool_calls = result.get("message", {}).get("tool_calls", [])
    if not tool_calls:
        return None, None

    call = tool_calls[0]
    function = call.get("function", {})
    name = function.get("name")
    arguments = function.get("arguments", {})

    return name, arguments


def handle_tool_call(result: dict):
    """
    Вызывает функцию из заранее определенного списка,
    в зависимости от того, что вернула LLM.

    :param result:
    :return:
    """
    name_, args_ = extract_tool_call_info(result)

    if name_ == "get_all_units":
        return get_all_units()
    elif name_ == "get_doctors_by_unit":
        return get_doctors_by_unit(args_["unit_name"])
    elif name_ == "get_service_info":
        # Пока не подключено...
        return get_service_info(args_["service_id"])
    else:
        return {"Ошибка": f"неизвестное название функции {name_}"}


if __name__ == "__main__":
    # question = input("Question: ")
    # result = call_ollama_with_functions(question)
    # print(result)
    # handle_tool_call(result)

    print("--------------------------")
    # print(get_all_units()[1]['name'])
    # unit_id = int(input("unit_id: "))
    units = get_all_units()
    for unit in units:
        print(unit)
    print("--------------------------")

    url = f"{base_url}/doctors"
    response = requests.get(url, auth=auth, verify=False)
    doctors = response.json()
    doctor_dict = {doc['id']: doc['fio'] for doc in doctors}

    # unit = next((u for u in units if u['id'] == unit_id), None)
    # if not unit:
    #     print(f"Ошибка: Направление {unit_id} не найдено.")
    # print(f"{unit=}")
    # unit_name = unit['name']
    # print(unit_name)

    url = f"{base_url}/doctorCompanyUnits"
    response = requests.get(url, auth=auth, verify=False)
    doctor_company_units = response.json()

    target_unit_id = 17

    matched = [
        {
            "worker": entry["worker"],
            "fio": doctor_dict.get(entry["worker"], "Не найдено"),
            "specialization": entry["specialization"]
        }
        for entry in doctor_company_units
        if entry.get("companyUnit") == target_unit_id
    ]


    # matched = [entry for entry in doctor_company_units if entry.get("companyUnit") == target_unit_id]
    matched1 = list(filter(lambda x: x.get("companyUnit") == 17, doctor_company_units))

    # Выводим результат
    for doctor in matched:
        print(doctor)
    print("--------------------------")
    for doctor in matched1:
        print(doctor)
