from pathlib import Path


def create_path(filename: str) -> str:
    current_file = Path(__file__).resolve()  # Полный путь к текущему модулю
    current_dir = current_file.parent  # Папка, где он лежит
    data_file = current_dir / "Upload" / filename

    # Превращаем Path-объект в строку:
    data_file_str = str(data_file)

    print("Путь к файлу:", data_file_str)

    return data_file_str

# Example:
# with data_file_path.open('r', encoding='utf-8') as f:
#     content = f.read()
#     ...
