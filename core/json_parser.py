"""
Improved JSON Parser with Robust Error Handling

Key improvements:
1. Refactored complex functions into smaller, manageable pieces
2. Simplified escaping using StringContext class
3. Improved nested code block handling with recursive approach
4. Added input validation for security
5. Prioritized parsing methods (fastest first)
6. Better code organization and documentation
"""

import json
import re
import ast
from typing import Any, Optional, Tuple
from enum import Enum


# ============================================================
# Constants and Enums
# ============================================================

class ParseStrategy(Enum):
    """Parsing strategies in priority order"""
    DIRECT = "direct"
    CODE_BLOCK_REMOVED = "code_block_removed"
    CONTROL_CHARS_ESCAPED = "control_chars_escaped"
    SINGLE_TO_DOUBLE_QUOTES = "single_to_double_quotes"
    COMBINED_ESCAPE_CONVERT = "combined_escape_convert"
    AST_LITERAL = "ast_literal"


# Maximum input size to prevent DoS (10MB)
MAX_INPUT_SIZE = 10 * 1024 * 1024

# Control characters that need escaping
CONTROL_CHAR_MAP = {
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
    '\b': '\\b',
    '\f': '\\f',
}


# ============================================================
# Input Validation
# ============================================================

def _validate_input(json_string: str) -> bool:
    """
    Validate input to prevent security vulnerabilities

    Args:
        json_string: Input string to validate

    Returns:
        True if valid, False otherwise
    """
    if not json_string or not isinstance(json_string, str):
        return False

    # Check size limit
    if len(json_string) > MAX_INPUT_SIZE:
        return False

    # Check for suspicious patterns (basic security check)
    # Prevent extremely deep nesting
    if json_string.count('{') > 1000 or json_string.count('[') > 1000:
        return False

    return True


# ============================================================
# Code Block Handling (Improved with Recursion)
# ============================================================

def _find_matching_code_block_end(lines: list, start_idx: int) -> Optional[int]:
    """
    Find the matching ``` for a code block start

    Args:
        lines: List of lines
        start_idx: Index of the opening ```

    Returns:
        Index of matching closing ``` or None
    """
    depth = 1
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith('```'):
            # Check if it's an opening or closing block
            # Opening: ```language
            # Closing: ``` (only)
            if re.match(r'^```\w+', line):
                depth += 1
            elif line == '```':
                depth -= 1
                if depth == 0:
                    return i
    return None


def _remove_outer_code_block(s: str) -> str:
    """
    Remove outermost code block only, preserving nested blocks

    Handles:
    - Simple blocks: ```json ... ```
    - Nested blocks: ```json ... ```inner``` ... ```

    Args:
        s: Input string

    Returns:
        String with outer code block removed
    """
    lines = s.strip().split('\n')

    if len(lines) < 2:
        return s.strip()

    first_line = lines[0].strip()

    # Check if first line is a code block marker
    if not re.match(r'^```\w*$', first_line):
        return s.strip()

    # Find matching closing block
    end_idx = _find_matching_code_block_end(lines, 0)

    if end_idx is not None:
        return '\n'.join(lines[1:end_idx])

    return s.strip()


# ============================================================
# String Processing Helpers
# ============================================================

class StringContext:
    """Track context while parsing strings"""
    def __init__(self):
        self.in_double_quote = False
        self.in_single_quote = False
        self.escape_next = False

    def is_in_string(self) -> bool:
        return self.in_double_quote or self.in_single_quote

    def process_char(self, char: str):
        """Update context based on character"""
        if self.escape_next:
            self.escape_next = False
            return

        if char == '\\':
            self.escape_next = True
        elif char == '"' and not self.in_single_quote:
            self.in_double_quote = not self.in_double_quote
        elif char == "'" and not self.in_double_quote:
            self.in_single_quote = not self.in_single_quote


def _escape_control_char(char: str) -> str:
    """
    Escape a single control character

    Args:
        char: Character to escape

    Returns:
        Escaped string representation
    """
    if char in CONTROL_CHAR_MAP:
        return CONTROL_CHAR_MAP[char]

    # Other control characters (0x00-0x1F)
    if ord(char) < 0x20:
        return f'\\u{ord(char):04x}'

    return char


def _escape_control_chars_in_strings(s: str) -> str:
    """
    Escape unescaped control characters inside JSON string values

    Improved version with StringContext for better tracking

    Args:
        s: Input string

    Returns:
        String with control characters escaped
    """
    result = []
    context = StringContext()

    for char in s:
        # Process character for context tracking
        was_escape = context.escape_next
        context.process_char(char)

        # Don't modify escaped characters
        if was_escape:
            result.append(char)
            continue

        # Don't modify escape character itself
        if char == '\\':
            result.append(char)
            continue

        # Don't modify quotes
        if char == '"' or char == "'":
            result.append(char)
            continue

        # Escape control characters only inside strings
        if context.is_in_string() and ord(char) < 0x20:
            result.append(_escape_control_char(char))
        else:
            result.append(char)

    return ''.join(result)


def _convert_single_to_double_quotes(s: str) -> str:
    """
    Convert Python-style single quotes to JSON double quotes

    Improved version with better quote handling

    Args:
        s: Input string

    Returns:
        String with single quotes converted to double quotes
    """
    result = []
    context = StringContext()

    for char in s:
        was_escape = context.escape_next
        was_in_single = context.in_single_quote
        was_in_double = context.in_double_quote

        context.process_char(char)

        # Handle escaped characters
        if was_escape:
            if char == "'" and not was_in_double:
                # Escaped single quote outside double-quoted string
                result.append("'")
            elif char == '"':
                # Escaped double quote
                result.append('\\"')
            else:
                result.append(char)
            continue

        # Handle escape character
        if char == '\\':
            result.append(char)
            continue

        # Handle quotes
        if char == '"':
            if not was_in_single:
                # Double quote outside single-quoted string
                result.append(char)
            else:
                # Double quote inside single-quoted string needs escaping
                result.append('\\"')
        elif char == "'":
            if not was_in_double:
                # Convert single quote to double quote
                result.append('"')
            else:
                # Single quote inside double-quoted string stays as is
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)


# ============================================================
# Parsing Strategies
# ============================================================

def _try_parse_direct(s: str) -> Tuple[Optional[Any], bool]:
    """Try parsing directly with json.loads"""
    try:
        return json.loads(s), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def _try_parse_with_code_block_removed(s: str) -> Tuple[Optional[Any], bool]:
    """Try parsing after removing code blocks"""
    try:
        cleaned = _remove_outer_code_block(s)
        return json.loads(cleaned), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def _try_parse_with_control_chars_escaped(s: str) -> Tuple[Optional[Any], bool]:
    """Try parsing after escaping control characters"""
    try:
        cleaned = _remove_outer_code_block(s)
        fixed = _escape_control_chars_in_strings(cleaned)
        return json.loads(fixed), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def _try_parse_with_quotes_converted(s: str) -> Tuple[Optional[Any], bool]:
    """Try parsing after converting single quotes to double quotes"""
    try:
        cleaned = _remove_outer_code_block(s)
        converted = _convert_single_to_double_quotes(cleaned)
        return json.loads(converted), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def _try_parse_with_combined_fixes(s: str) -> Tuple[Optional[Any], bool]:
    """Try parsing after applying both quote conversion and control char escaping"""
    try:
        cleaned = _remove_outer_code_block(s)
        converted = _convert_single_to_double_quotes(cleaned)
        fixed = _escape_control_chars_in_strings(converted)
        return json.loads(fixed), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def _try_parse_with_ast(s: str) -> Tuple[Optional[Any], bool]:
    """Try parsing with ast.literal_eval as last resort"""
    for text in [s, _remove_outer_code_block(s)]:
        try:
            result = ast.literal_eval(text.strip())
            return result, True
        except (ValueError, SyntaxError):
            continue
    return None, False


# ============================================================
# Main Parser
# ============================================================

def parse_json_robust(json_string: str) -> Optional[Any]:
    """
    Robustly parse JSON strings with various formatting issues

    Features:
    - Standard JSON
    - Code blocks (```json ... ```)
    - Nested code blocks
    - Unescaped control characters (\n, \t, etc.)
    - Python-style dictionaries (single quotes)
    - Mixed special characters

    Improvements:
    - Prioritized parsing strategies (fastest first)
    - Better nested code block handling
    - Input validation for security
    - Refactored into smaller functions

    Args:
        json_string: JSON string to parse

    Returns:
        Parsed Python object (dict, list, etc.) or None on failure
    """
    # Input validation
    if not _validate_input(json_string):
        return None

    # Try parsing strategies in order of priority (fastest first)
    strategies = [
        _try_parse_direct,
        _try_parse_with_code_block_removed,
        _try_parse_with_control_chars_escaped,
        _try_parse_with_quotes_converted,
        _try_parse_with_combined_fixes,
        _try_parse_with_ast,
    ]

    for strategy in strategies:
        result, success = strategy(json_string)
        if success:
            return result

    return None


def parse_json_strict(json_string: str) -> Any:
    """
    Parse JSON and raise exception on failure

    Args:
        json_string: JSON string to parse

    Returns:
        Parsed Python object

    Raises:
        ValueError: If parsing fails
    """
    result = parse_json_robust(json_string)
    if result is None:
        preview = json_string[:100] + '...' if len(json_string) > 100 else json_string
        raise ValueError(f"JSON parsing failed: {preview}")
    return result
