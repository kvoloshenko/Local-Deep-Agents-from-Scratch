"""State management for deep agents with TODO tracking and virtual file systems.

Этот модуль определяет расширенную структуру состояния агента (state) для Deep Agents.

Состояние хранится между шагами графа (LangGraph) и используется для:
- отслеживания хода выполнения задач через TODO-список,
- offloading контекста через виртуальную файловую систему (files),
- эффективного объединения состояний агента с помощью редьюсеров (reducers).

DeepAgentState — это фундаментальная часть Deep Agents, позволяющая:
- сохранять план выполнения,
- сохранять промежуточные результаты,
- делать агента более устойчивым на длинных траекториях.
"""

from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

# В LangChain 1.0 AgentState переместили в langchain.agents
# Это базовый state, который содержит как минимум поле messages и служебные структуры.
from langchain.agents import AgentState  


class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Структурированный элемент TODO-списка — используется агентом для планирования.

    Каждый TODO описывает одно действие:
        content: текст задачи, короткий и конкретный
        status: статус выполнения — pending / in_progress / completed

    Такой формат обеспечивает:
    - машиночитаемость,
    - возможность сохранять список в состоянии,
    - возможность обновлять статус через write_todos().
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence.

    Редьюсер для поля files — определяет правила объединения содержимого
    "виртуальной файловой системы" между шагами выполнения.

    В LangGraph редьюсеры позволяют задавать, как объединять новое состояние
    с предыдущим. Для файлов — это особенно важно, т.к. они часто обновляются.

    Принцип работы:
        - left  — текущее состояние files
        - right — новые изменения (новые/обновлённые файлы)
        - при конфликте правое значение (right) перезаписывает левое

    Это делает файловую систему "всегда актуальной", как обычный dict.
    """

    # Если старого значения нет — возвращаем новое
    if left is None:
        return right
    # Если нет нового — оставляем старое
    elif right is None:
        return left
    # Если оба существуют — перезаписываем пересекающиеся ключи значениями right
    else:
        return {**left, **right}


class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Расширенное состояние агента Deep Agents.

    Наследуется от базового AgentState, который уже содержит:
    - messages — список сообщений (Human, AI, ToolMessage),
    - служебные поля выполнения графа.

    Добавляем два ключевых поля:

    1) todos — список задач (план работы)
       Может отсутствовать в начале, поэтому используется NotRequired.
       Агент создаёт и обновляет TODO-список с помощью инструмента write_todos().

    2) files — виртуальная файловая система:
       dict[str, str] = { "filename.md": "file content" }
       Используется для:
       - сохранения сырых результатов поиска (raw content),
       - хранения промежуточных данных,
       - offloading объёмной информации за пределы context window.

       К этому полю подключён редьюсер file_reducer, который обеспечивает
       корректное объединение файлов при каждом шаге графа.

    Таким образом, DeepAgentState служит "памятью" агента:
    - краткосрочной (messages),
    - структурированной (todos),
    - долговременной / объёмной (files).
    """

    # TODO-список. Может отсутствовать → создаётся инструментом write_todos
    todos: NotRequired[list[Todo]]

    # Виртуальная файловая система — dict[file_path → content]
    # Annotated позволяет прикрепить редьюсер к полю.
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
