# audio_preparation

Скрипт для подготовки аудио: извлекает реальные utt и генерирует фейковые через TTS.

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
- Убедись, что папка `voxforge_raw` присутствует и имеет структуру проекта VoxForge.
- Некоторые TTS-модели (Silero, Coqui) загружаются из интернета при первом запуске.
- Если загрузка модели не удалась, скрипт пропустит соответствующий backend и продолжит работу.
