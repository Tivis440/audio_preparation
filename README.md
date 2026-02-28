# audio_preparation

Скрипт для подготовки аудио: автоматически скачивает архивы с VoxForge, извлекает аудио, обрабатывает и генерирует fake audio через TTS.

## Быстрый старт в Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tivis440/audio_preparation/blob/main/colab_notebook.ipynb)

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

Инициализация репозитория и пуш на GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
# создать репозиторий на GitHub вручную или через gh/API, затем:
git remote add origin git@github.com:<user>/<repo>.git
git branch -M main
git push -u origin main
```

Примечания:
- Скрипт автоматически качает архивы с VoxForge (можно настроить `MAX_FILES` и `MIN_SPEAKERS`)
- TTS-модели (Silero, Coqui) загружаются из интернета при первом запуске
- Если загрузка модели не удалась, скрипт пропустит соответствующий backend и продолжит работу
- В результате создаются папки `data/real` и `data/fake/<engine>` с audio файлами
- `metadata.csv` содержит информацию об уникальных высказываниях (utt_id, speaker, text)
