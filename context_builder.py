def build_context(context_blocks):
    formatted = ""
    for block in context_blocks:
        date_str = block.get("published", "unknown date")
        formatted += (
            f"From {block['channel']} - {block['title']} (uploaded {date_str}) [{block['start']}s]:\n"
            f"{block['text']}\n\n"
        )
    return formatted.strip()
