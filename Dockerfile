# Базовый образ Python
FROM python:3.12.3

# Установка рабочей директории
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Открытие порта для Gradio
EXPOSE 7860

# Команда для запуска приложения
CMD ["python", "gradio_interface.py"]