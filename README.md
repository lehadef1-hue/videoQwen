# videoQwen

Автоматический анализатор видео на базе [Qwen3-VL-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8), запущенного локально через vLLM на RunPod.

## Что делает

- Извлекает теги/категории, ориентацию, описание из видеофайлов
- Определяет студию/водяной знак
- Находит лучшие кадры (thumbnail + галерея) по числовой шкале качества
- Опционально идентифицирует актёров по лицам (InsightFace + своя база)
- Выдаёт HTML-отчёт с пагинацией, лайтбоксом и тёмной темой

## Архитектура

```
┌────────────────────┐   POST /generate   ┌──────────────────────────────┐
│  video_processor   │ ────────────────►  │  model_server                │
│  FastAPI :8000     │ ◄────────────────  │  vLLM + Qwen3-VL-30B  :8080  │
└────────────────────┘    JSON response   └──────────────────────────────┘
         │
         ▼
   result/<run_name>/
     <video>/
       *_meta.json      ← теги, студия, актёры
       thumbnail.jpg
       frame_*.jpg
   report.html          ← финальный отчёт
```

### 4-проходный конвейер на видео

| Проход | Кадры | Цель |
|--------|-------|------|
| 1a | 25 кадров | ориентация + описание + теги + студия |
| 2a | первая половина, 25 кадров | дополнительные теги |
| 2b | вторая половина, 25 кадров | дополнительные теги |

Первые/последние 4–8% видео пропускаются (интро/аутро).

## Быстрый старт (RunPod)

### 1. Клонировать репозиторий

```bash
cd /workspace
git clone https://github.com/lehadef1-hue/videoQwen videoQwen
cd videoQwen
```

### 2. Развернуть окружение

```bash
bash setup.sh
```

Скрипт создаёт `.venv` внутри проекта, устанавливает vllm (latest) и зависимости.
Модель скачивается автоматически при первом запуске model_server в `/workspace/hf_cache`.

### 3. Запустить сервисы (в двух терминалах)

```bash
# Терминал 1 — модель (грузится ~2–3 мин)
bash start_model_server.sh

# Терминал 2 — приложение (после того как модель поднялась)
bash start_app.sh
```

Приложение доступно на `http://localhost:8000`.

### 4. Запустить анализ

Через веб-интерфейс: укажи папку с видео и имя запуска, нажми **Start**.

Или через API:

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"video_dir": "/workspace/videos", "run_name": "my_run"}'
```

Результат: `result/my_run/report.html`.

## Опциональный поиск актёров

Требует отдельной установки и заполненной базы лиц.

### Установка InsightFace

```bash
source .venv/bin/activate
pip install insightface onnxruntime-gpu
```

### Создание базы исполнителей

1. Зарегистрироваться на [theporndb.net](https://theporndb.net/register) (бесплатно)
2. Получить API-токен в личном кабинете

```bash
export TPDB_API_TOKEN=your_token_here

# Добавить конкретных актёров
python build_performer_db.py --names "Name One" "Name Two"

# Или авто-заполнение топ-200 по популярности
python build_performer_db.py --auto --count 200

# Посмотреть содержимое базы
python build_performer_db.py --list

# Удалить запись
python build_performer_db.py --remove "Wrong Name"
```

База сохраняется в `performers_db.pkl`. Путь можно переопределить:

```bash
export PERFORMER_DB_PATH=/workspace/my_performers.pkl
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `HF_HOME` | `/workspace/hf_cache` | Кэш HuggingFace (задан в model_server.py) |
| `PERFORMER_DB_PATH` | `performers_db.pkl` | База лиц исполнителей |
| `PERFORMER_THRESHOLD` | `0.42` | Порог совпадения лица (0–1) |
| `TPDB_API_TOKEN` | — | Токен ThePornDB для build_performer_db.py |

## Структура файлов

```
videoQwen/
├── model_server.py          # vLLM сервер (порт 8080)
├── video_processor.py       # FastAPI приложение (порт 8000)
├── performer_finder.py      # Модуль распознавания лиц
├── build_performer_db.py    # CLI для сборки базы исполнителей
├── setup.sh                 # Скрипт развёртывания
├── start_model_server.sh    # Запуск model_server (генерируется setup.sh)
├── start_app.sh             # Запуск video_processor (генерируется setup.sh)
├── templates/
│   └── index.html           # Шаблон HTML-отчёта
├── result/                  # Результаты анализа (в .gitignore)
└── performers_db.pkl        # База лиц (в .gitignore)
```

## Зависимости

- Python 3.10+
- CUDA GPU (минимум ~40 GB VRAM для Qwen3-VL-30B-FP8)
- vllm
- fastapi, uvicorn
- opencv-python-headless, pillow
- transformers, torch
- requests, jinja2, numpy
- *(опц.)* insightface, onnxruntime-gpu
