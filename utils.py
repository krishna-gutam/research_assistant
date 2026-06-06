def sanitize_content(content) -> str:
    """Safely extracts a string from LangChain's potential list-based content blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, str):
                extracted.append(block)
            elif isinstance(block, dict) and "text" in block:
                extracted.append(block["text"])
        return "\n".join(extracted)
    return str(content)