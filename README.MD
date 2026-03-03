# ComfyUI Qwen3.5-VL Node

A custom implementation of the Qwen 3.5 Vision-Language model for the ComfyUI framework. This extension allows for text, image, and video inference using Qwen 3.5 architectures.

## Architecture and Dependency Management
The Qwen 3.5 models require `transformers==5.2.0`. Updating the global `transformers` library within the ComfyUI environment typically causes version conflicts with other custom nodes relying on `transformers 4.x`. 

To resolve this, this node implements a **vendoring** approach with dynamic module switching. Dependencies (`transformers`, `huggingface_hub`, `tokenizers`) are installed in an isolated local directory (`vendor/`). A context manager temporarily patches `sys.modules` during initialization and inference, ensuring the 5.x execution context remains completely separated from the global ComfyUI environment.

## Features
- Support for Qwen 3.5 model sizes (0.8B, 2B, 4B, 9B).
- Multi-modal inference (Text, Single Image, Video/Image Batch).
- Hardware optimization: 4-bit quantization support (`BitsAndBytesConfig`), `torch.compile` compatibility, and configurable attention modes (`sdpa`, `flash_attention_2`, `eager`).
- Automated dependency fetching on initial startup.

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
```
2. Clone this repository:
```bash
git clone <repository_url>
```
3. Start ComfyUI. The extension will automatically download the required isolated dependencies into the `vendor` directory on the first run. 

*Note: Initial startup may take longer than usual while the dependencies are being fetched.*

## Usage
The node will appear in the ComfyUI menu under the **Qwen3.5** category.
1. Connect an image or video output to the `image_or_video` input (optional).
2. Specify the `model_size` and your prompt.
3. Configure performance parameters (e.g., `use_4bit`, `attention_mode`).

Models will be downloaded automatically from the Hugging Face Hub to `ComfyUI/models/qwen35` upon first execution.

---

# ComfyUI Qwen3.5-VL Node (RU)

Кастомная нода для интеграции Vision-Language модели Qwen 3.5 в среду ComfyUI. Дополнение обеспечивает поддержку инференса текста, изображений и видео.

## Архитектура и управление зависимостями
Для работы моделей Qwen 3.5 требуется библиотека `transformers==5.2.0`. Обновление `transformers` в глобальном окружении ComfyUI приводит к конфликту версий с другими нодами, требующими ветку 4.x.

Для решения этой проблемы применяется метод **vendoring** (изоляция зависимостей) с динамическим переключением модулей. Требуемые пакеты устанавливаются в локальную директорию (`vendor/`). Контекстный менеджер временно подменяет `sys.modules` во время инициализации и генерации, гарантируя полную изоляцию контекста выполнения 5.x от глобального окружения ComfyUI.

## Характеристики
- Поддержка параметров Qwen 3.5 (0.8B, 2B, 4B, 9B).
- Мультимодальный инференс (Текст, Изображение, Батчи изображений/Видео).
- Аппаратные оптимизации: поддержка 4-битного квантования, совместимость с `torch.compile`, выбор режимов внимания (`sdpa`, `flash_attention_2`, `eager`).
- Автоматическая загрузка изолированных зависимостей при первом запуске.

## Установка

1. Перейдите в директорию пользовательских нод ComfyUI:
```bash
cd ComfyUI/custom_nodes
```
2. Клонируйте репозиторий:
```bash
git clone <repository_url>
```
3. Запустите ComfyUI. Дополнение автоматически скачает необходимые зависимости в директорию `vendor` при первой инициализации.

*Примечание: Первый запуск может занять больше времени из-за процесса установки пакетов.*

## Использование
Нода доступна в меню ComfyUI в категории **Qwen3.5**.
1. Подключите изображение или видео к входу `image_or_video` (опционально).
2. Задайте `model_size` и текстовый промпт.
3. Настройте параметры производительности (`use_4bit`, `attention_mode`).

Веса моделей загружаются автоматически из Hugging Face Hub в директорию `ComfyUI/models/qwen35` при первом выполнении.