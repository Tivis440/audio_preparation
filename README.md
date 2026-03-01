# audio_preparation

Скрипт для подготовки аудио: автоматически скачивает архивы с VoxForge, извлекает аудио, обрабатывает и генерирует fake audio через TTS.

## Быстрый старт в Google Colab

Откройте упрощённый ноутбук, который устанавливает зависимости из
`requirements.txt` и включает последние исправления:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tivis440/audio_preparation/blob/main/run_in_colab.ipynb)

Просто откройте ссылку выше и запустите ячейки по порядку.

## Локальный запуск

Установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запуск (пример):

```bash
python main.py
```


Примечания:
- Скрипт автоматически качает архивы с VoxForge (можно настроить `MAX_FILES` и `MIN_SPEAKERS`)
- TTS-модели (Silero, Coqui) загружаются из интернета при первом запуске
- Если загрузка модели не удалась, скрипт пропустит соответствующий backend и продолжит работу
- В результате создаются папки `data/real` и `data/fake/<engine>` с audio файлами
- `metadata.csv` содержит информацию об уникальных высказываниях (utt_id, speaker, text)

В репозитории также есть `run_one_archive.py` — вспомогательный скрипт
для быстрого теста одного архива; он автоматически пропускает пакеты без
расшифровок.
