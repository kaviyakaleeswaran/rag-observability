def extract_citations(text):
    """
    Extract valid citation IDs from an LLM response.

    Supported formats:
    - resnet_38
    - - resnet_38
    - * resnet_38

    Returns:
        list of valid citation IDs
    """

    valid_sources = {
        "resnet",
        "faster_rcnn",
        "unet",
        "yolo",
        "vit"
    }

    citations = []

    for line in text.splitlines():
        line = line.strip()

        # Remove bullet markers
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()

        # Check citation format: source_number
        if "_" not in line:
            continue

        parts = line.rsplit("_", 1)

        if len(parts) != 2:
            continue

        source, number = parts

        if source in valid_sources and number.isdigit():
            citation = f"{source}_{number}"

            if citation not in citations:
                citations.append(citation)

    return citations