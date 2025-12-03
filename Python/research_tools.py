"""Research Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.

Этот модуль содержит утилиты для:
- веб-поиска (через Tavily),
- обработки и суммаризации содержимого страниц,
- сохранения результатов в файлы (для offloading контекста).

Он используется исследовательским агентом (research agent) в Deep Agents.
"""
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid, base64  # uuid и base64 используются для генерации уникальных имён файлов

import httpx
# from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify  # конвертация HTML → Markdown
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

# from deep_agents_from_scratch.prompts import SUMMARIZE_WEB_SEARCH
from prompts import SUMMARIZE_WEB_SEARCH
# from deep_agents_from_scratch.state import DeepAgentState
from state import DeepAgentState

# Загружаем переменные окружения из файла .env
load_dotenv()

# Считываем переменные окружения, необходимые для LangSmith/LangChain
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LLM = os.getenv("LLM")

# ---------------------------------------------------------------------------
# 🧠 МОДЕЛЬ ДЛЯ СУММАРИЗАЦИИ
# ---------------------------------------------------------------------------
# Отдельная LLM-модель, которая будет использоваться для структурированной
# суммаризации содержимого страниц. Здесь используется openai:gpt-4o-mini
# через init_chat_model (LangChain).
# summarization_model = init_chat_model(model="openai:gpt-4o-mini")
summarization_model = ChatOllama(
    model=LLM,
    # опционально:
    # num_ctx=8192,
    # temperature=0.2,
)

# Клиент Tavily — внешний API для интеллектуального веб-поиска
tavily_client = TavilyClient()


class Summary(BaseModel):
    """Schema for webpage content summarization.

    Pydantic-модель, описывающая структуру ответа суммаризатора.
    Используется с with_structured_output, чтобы LLM возвращала
    строго заданный JSON-формат: имя файла + краткое summary.
    """
    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


# def get_today_str() -> str:
#     """Get current date in a human-readable format.
#
#     Возвращает текущую дату в удобочитаемом формате, который
#     подставляется в промпты (например, при суммаризации).
#     """
#     return datetime.now().strftime("%a %b %-d, %Y")
from datetime import datetime
def get_today_str() -> str:
    """Get current date in a human-readable, cross-platform format."""
    # "%a %b %d, %Y" даёт, например: "Mon Dec 01, 2025"
    # Далее убираем ведущий ноль у дня месяца
    raw = datetime.now().strftime("%a %b %d, %Y")
    # "Dec 01" -> "Dec 1"
    return raw.replace(" 0", " ")

def run_tavily_search(
    search_query: str, 
    max_results: int = 1, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> dict:
    """Perform search using Tavily API for a single query.

    Выполняет один запрос к Tavily API.

    Args:
        search_query: поисковый запрос
        max_results: максимальное количество результатов
        topic: тема (общий, новости, финансы)
        include_raw_content: включать ли сырое содержимое страниц

    Returns:
        Словарь результатов Tavily (JSON → dict).
    """
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )

    return result


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model.

    Суммаризирует содержимое страницы с использованием настроенной LLM-модели
    и структурированного вывода в формате Summary.

    Args:
        webpage_content: сырое содержимое страницы (обычно markdown)

    Returns:
        Объект Summary с filename и summary.
    """
    try:
        # Настраиваем модель на структурированный вывод согласно схеме Summary
        structured_model = summarization_model.with_structured_output(Summary)

        # Формируем промпт: подставляем содержимое страницы и дату в шаблон
        summary_and_filename = structured_model.invoke([
            HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])

        # Модель возвращает Summary (filename + summary)
        return summary_and_filename

    except Exception:
        # На случай любой ошибки — возвращаем запасной Summary,
        # где summary = первые 1000 символов текста (или всё, если короче)
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
        )


def process_search_results(results: dict) -> list[dict]:
    """Process search results by summarizing content where available.

    Обрабатывает результаты Tavily-поиска:
    - пытается скачать HTML по URL,
    - конвертирует HTML → markdown,
    - суммаризирует содержимое,
    - генерирует уникальное имя файла,
    - возвращает список структурированных результатов.

    Args:
        results: словарь с результатами поиска Tavily

    Returns:
        Список словарей: url, title, summary, filename, raw_content.
    """
    processed_results = []

    # Отдельный HTTP-клиент с таймаутом — чтобы не зависнуть на долгих запросах
    HTTPX_CLIENT = httpx.Client(timeout=30.0)  # таймаут 30 секунд

    # Итерируемся по списку результатов Tavily
    for result in results.get('results', []):

        # Извлекаем URL
        url = result['url']

        # Пытаемся считать страницу по URL
        try:
            response = HTTPX_CLIENT.get(url)

            if response.status_code == 200:
                # Если всё ок — конвертируем HTML → markdown
                raw_content = markdownify(response.text)
                # Суммаризируем markdown-содержимое
                summary_obj = summarize_webpage_content(raw_content)
            else:
                # Если код ответа не 200 — fallback:
                # используем сырое содержимое/summary от Tavily
                raw_content = result.get('raw_content', '')
                summary_obj = Summary(
                    filename="URL_error.md",
                    summary=result.get('content', 'Error reading URL; try another search.')
                )
        except (httpx.TimeoutException, httpx.RequestError) as e:
            # Обработка ошибок соединения/таймаута — не ломаем пайплайн,
            # а возвращаем понятное сообщение и используем Tavily content
            raw_content = result.get('raw_content', '')
            summary_obj = Summary(
                filename="connection_error.md",
                summary=result.get('content', f'Could not fetch URL (timeout/connection error). Try another search.')
            )

        # Генерируем уникальный суффикс для имени файла (чтобы не было коллизий)
        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"

        # Собираем структуру результата
        processed_results.append({
            'url': result['url'],
            'title': result['title'],
            'summary': summary_obj.summary,
            'filename': summary_obj.filename,
            'raw_content': raw_content,
        })

    return processed_results


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Выполняет веб-поиск и сохраняет подробные результаты в виртуальные файлы
    (для offloading контекста). В контекст (messages) возвращает только краткую
    сводку: какие файлы созданы и что в них в целом содержится.

    Args:
        query: поисковой запрос
        state: состояние агента (InjectedState), нужно для доступа к files
        tool_call_id: идентификатор вызова инструмента (для ToolMessage)
        max_results: максимальное количество результатов
        topic: тип поиска ('general' | 'news' | 'finance')

    Returns:
        Command, который:
        - обновляет files (создаёт файлы с результатами поиска),
        - добавляет ToolMessage с кратким summary.
    """
    # 1. Выполняем Tavily-поиск
    search_results = run_tavily_search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    ) 

    # 2. Обрабатываем и суммаризируем каждый результат
    processed_results = process_search_results(search_results)

    # 3. Подготовка обновлений файлов и краткой сводки
    files = state.get("files", {})
    saved_files = []
    summaries = []

    for i, result in enumerate(processed_results):
        # Используем имя файла, возвращённое суммаризатором
        filename = result['filename']

        # Формируем содержимое файла:
        # - заголовок
        # - URL
        # - исходный запрос
        # - дата
        # - краткое summary
        # - сырое содержимое (markdown)
        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""

        # Сохраняем в виртуальную файловую систему (state["files"])
        files[filename] = file_content
        saved_files.append(filename)
        # Для краткой сводки добавляем одну строку по каждому файлу
        summaries.append(f"- {filename}: {result['summary']}...")

    # 4. Краткое текстовое summary для ToolMessage — чтобы агент понимал:
    # - сколько результатов найдено,
    # - как они примерно выглядят,
    # - имена файлов, которые можно читать через read_file().
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

    # 5. Возвращаем Command — LangGraph применит обновления к state.
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(summary_text, tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Инструмент для стратегической рефлексии в процессе исследования.

    Зачем нужен:
    - создать «паузу на подумать» между вызовами поисковых инструментов;
    - явно формулировать:
        * что уже найдено,
        * чего не хватает,
        * достаточно ли данных,
        * стоит ли продолжать поиск или уже отвечать.

    Args:
        reflection: развёрнутая мысль агента о ходе исследования

    Returns:
        Строка-подтверждение, что рефлексия «записана».
        (Важно для логов и понимания человеком, как мыслит агент.)
    """
    return f"Reflection recorded: {reflection}"
