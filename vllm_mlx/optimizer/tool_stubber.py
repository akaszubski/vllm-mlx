# SPDX-License-Identifier: Apache-2.0
"""
Tool description stubber + JSON schema simplifier.

Replaces verbose tool descriptions with short stubs (Claude Code tool name ->
short stub) and recursively strips non-structural metadata from input_schema
while preserving the fields the model actually needs to call the tool.

The STUBS table below is hand-maintained against current Claude Code tool
names. Tools with no stub entry keep their original description (the schema
simplifier still runs). The optimizer logs a one-time warning when it sees
an unknown tool name so drift is visible.
"""

from typing import Any

from ..api.anthropic_models import AnthropicToolDef

# STUBS captured against Claude Code v2.x (last reviewed 2026-04). Update
# this table when Anthropic ships new tools or renames existing ones; the
# test suite pins the key set so renames break CI.
STUBS: dict[str, str] = {
    "Bash": "Executes bash commands with optional timeout.",
    "Read": "Reads a file by path with optional offset/limit.",
    "Write": "Writes content to a file path.",
    "Edit": "Replaces old_string with new_string in a file.",
    "Glob": "Finds files matching a glob pattern.",
    "Grep": "Searches for regex pattern in files.",
    "Task": "Launches a subagent for complex tasks.",
    "TodoWrite": "Creates and manages a task list.",
    "ExitPlanMode": "Exits plan mode for user approval.",
    "NotebookEdit": "Edits Jupyter notebook cells.",
    "WebFetch": "Fetches and processes content from a URL.",
    "WebSearch": "Searches the web for information.",
    "BashOutput": "Gets output from a background shell.",
    "KillShell": "Stops a running background shell.",
    "Skill": "Executes a named skill in the conversation.",
    "SlashCommand": "Executes a slash command.",
    "TaskCreate": "Creates a task in the task list.",
    "TaskUpdate": "Updates a task status or details.",
    "TaskGet": "Retrieves a task by ID.",
    "TaskList": "Lists all tasks.",
    "TaskOutput": "Gets output from a background task.",
    "TaskStop": "Stops a running background task.",
    "AskUserQuestion": "Asks the user a question during execution.",
    "EnterPlanMode": "Transitions into plan mode for planning.",
}

_KEEP_KEYS = (
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "const",
    "additionalProperties",
    "anyOf",
    "oneOf",
    "allOf",
    "format",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "default",
    "propertyNames",
    "$schema",
)
_RECURSE_OBJECT_KEYS = ("items", "additionalProperties", "propertyNames")
_RECURSE_LIST_KEYS = ("anyOf", "oneOf", "allOf")


def simplify_schema(schema: Any) -> Any:
    """Recursively keep only structural JSON Schema fields.

    Strips ``description``, ``examples``, ``title``, ``$comment``, and any
    other non-structural metadata. Preserves nested schemas under
    ``properties``, ``items``, ``additionalProperties``, ``propertyNames``,
    and the ``anyOf``/``oneOf``/``allOf`` combinators.
    """
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [simplify_schema(item) for item in schema]
        return schema

    result: dict[str, Any] = {}
    for key in _KEEP_KEYS:
        if key not in schema:
            continue
        value = schema[key]
        if key == "properties" and isinstance(value, dict):
            result[key] = {p: simplify_schema(v) for p, v in value.items()}
        elif key in _RECURSE_OBJECT_KEYS and isinstance(value, dict):
            result[key] = simplify_schema(value)
        elif key in _RECURSE_LIST_KEYS and isinstance(value, list):
            result[key] = [simplify_schema(v) for v in value]
        else:
            result[key] = value
    return result


def stub_tools(
    tools: list[AnthropicToolDef],
) -> tuple[list[AnthropicToolDef], int]:
    """Replace verbose descriptions with stubs and simplify schemas.

    Only replaces a description when:
      - the tool name has an entry in ``STUBS``, AND
      - the existing description is longer than the stub.

    Always simplifies ``input_schema`` (the larger token win on most tools).

    Returns ``(new_tools, count_with_stubbed_descriptions)``.
    """
    new_tools: list[AnthropicToolDef] = []
    stubbed = 0
    for tool in tools:
        stub = STUBS.get(tool.name)
        new_description = tool.description
        if (
            stub is not None
            and tool.description is not None
            and len(tool.description) > len(stub)
        ):
            new_description = stub
            stubbed += 1

        new_schema = (
            simplify_schema(tool.input_schema)
            if tool.input_schema is not None
            else tool.input_schema
        )

        new_tools.append(
            tool.model_copy(
                update={
                    "description": new_description,
                    "input_schema": new_schema,
                }
            )
        )
    return new_tools, stubbed
