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
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify  # конвертация HTML → Markdown
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

from prompts import SUMMARIZE_WEB_SEARCH
from state import DeepAgentState

# Загружаем переменные окружения из файла .env
load_dotenv()

# Считываем переменные окружения, необходимые для Tavily/LangSmith/LangChain
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LLM = os.getenv("LLM")

# ---------------------------------------------------------------------------
#  МОДЕЛЬ ДЛЯ СУММАРИЗАЦИИ
# ---------------------------------------------------------------------------
# Отдельная LLM-модель, которая будет использоваться для структурированной
# суммаризации содержимого страниц.
summarization_model = ChatOllama(
    model=LLM,
    # опционально:
    # num_ctx=8192,
    # temperature=0.2,
)

# Клиент Tavily — внешний API для интеллектуального веб-поиска
tavily_client = TavilyClient()


# ---------------------------------------------------------------------------
#  УТИЛИТА: ПЕРЕВОД ПОИСКОВОГО ЗАПРОСА НА АНГЛИЙСКИЙ
# ---------------------------------------------------------------------------
def translate_query_to_english(query: str) -> str:
    """Переводит (или улучшает) поисковый запрос для веб-поиска на английском.

    Требования:
    - Возвращает ТОЛЬКО строку запроса (без кавычек, пояснений, маркдауна).
    - Сохраняет имена, бренды, версии, аббревиатуры.
    - Если запрос уже на английском, может слегка нормализовать, но не менять смысл.
    """
    # Важно: это *внутренний* перевод для улучшения качества поиска.
    # Итоговые ответы и summary всё равно остаются на русском.
    prompt = (
        "Translate the following search query to English for web search. "
        "Return ONLY the translated query as a single line, no quotes, no extra text.\n\n"
        f"Query: {query}"
    )
    try:
        resp = summarization_model.invoke([HumanMessage(content=prompt)])
        translated = (resp.content or "").strip()
        return translated if translated else query
    except Exception:
        return query


class Summary(BaseModel):
    """Schema for webpage content summarization.

    Pydantic-модель, описывающая структуру ответа суммаризатора.
    Используется с with_structured_output, чтобы LLM возвращала
    строго заданный JSON-формат: имя файла + краткое summary.
    """
    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def get_today_str() -> str:
    """Get current date in a human-readable, cross-platform format."""
    raw = datetime.now().strftime("%a %b %d, %Y")
    # "Dec 01" -> "Dec 1"
    return raw.replace(" 0", " ")


def run_tavily_search(
    search_query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Wrapper around tavily_client.search(...) to keep callsite clean."""
    return tavily_client.search(
        query=search_query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize page content and generate filename using structured output."""
    try:
        structured = summarization_model.with_structured_output(Summary)
        prompt = SUMMARIZE_WEB_SEARCH.format(
            webpage_content=webpage_content,
            date=get_today_str(),
        )
        result: Summary = structured.invoke([HumanMessage(content=prompt)])
        return result
    except Exception:
        # fallback: если structured output не сработал
        trimmed = webpage_content.strip()
        if len(trimmed) > 800:
            trimmed = trimmed[:800] + "..."
        return Summary(filename="search_result.md", summary=trimmed)


def process_search_results(results: dict) -> list[dict]:
    """Download content for each result, convert to markdown, summarize."""
    processed = []
    for r in results.get("results", []):
        url = r.get("url")
        title = r.get("title") or url
        raw_content = ""

        # Если Tavily вернул raw_content — используем его
        if r.get("raw_content"):
            raw_content = r["raw_content"]
        else:
            # Иначе пытаемся скачать страницу сами
            try:
                with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    raw_content = resp.text
            except Exception:
                raw_content = ""

        md = markdownify(raw_content) if raw_content else ""
        summary_obj = summarize_webpage_content(md or raw_content or title)

        # генерируем уникальное имя файла (чтобы не перетирать)
        unique = uuid.uuid4().hex
        safe_title = base64.urlsafe_b64encode(unique.encode()).decode()[:12]
        filename = summary_obj.filename
        if not filename or not filename.endswith(".md"):
            filename = f"search_{safe_title}.md"

        processed.append(
            {
                "url": url,
                "title": title,
                "summary": summary_obj.summary,
                "filename": filename,
                "raw_content": md,
            }
        )
    return processed


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
        query: поисковой запрос (может быть на русском)
        state: состояние агента (InjectedState),
        tool_call_id: технический ID вызова инструмента,
        max_results: сколько результатов взять (по умолчанию 1),
        topic: тип поиска ('general' | 'news' | 'finance').

    Returns:
        Command, который:
        - обновляет files (создаёт файлы с результатами поиска),
        - добавляет ToolMessage с кратким summary.
    """
    # 1) Переводим запрос на английский (для максимального recall в веб-поиске)
    english_query = translate_query_to_english(query)

    # 2) Запускаем Tavily-поиск (на английском)
    search_results = run_tavily_search(
        english_query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 3) Обрабатываем результаты (скачивание/markdownify/summary)
    processed_results = process_search_results(search_results)

    # 4) Сохраняем каждый результат в виртуальные файлы
    new_files = dict(state.get("files", {}))
    for result in processed_results:
        content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query (original):** {query}
**Query (english):** {english_query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content']}
"""
        new_files[result["filename"]] = content

    # 5) Формируем минимальный текст для ToolMessage (в контекст)
    lines = []
    for r in processed_results:
        lines.append(f"- {r['filename']}: {r['summary']}")

    summary_text = (
        f"Found {len(processed_results)} result(s). "
        f"Saved to files:\n" + "\n".join(lines)
    )

    return Command(
        update={
            "files": new_files,
            "messages": [
                ToolMessage(content=summary_text, tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def think_tool(reflection: str) -> str:
    """No-op инструмент для управляемой рефлексии."""
    return f"Reflection recorded: {reflection}"
