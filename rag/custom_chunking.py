from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class PolicyChunk:
    policy_title: str
    section_title: str
    text: str
    chunk_id: str
    chunk_index_in_section: int
    block_type: str  # "text" or "table"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_policy_title(markdown_text: str, file_path: str) -> str:
    """
    Use first H1 as policy title.
    Fallback to file stem if no H1 exists.
    """
    for line in markdown_text.splitlines():
        line = line.strip()
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()

    return Path(file_path).stem.replace("_", " ").replace("-", " ").strip()


def split_markdown_into_sections(markdown_text: str) -> List[tuple[str, str]]:
    """
    Split on H2+ headings only.
    Ignore H1 because it becomes policy_title.
    """
    lines = markdown_text.splitlines()

    sections: List[tuple[str, List[str]]] = []
    current_heading = "Introduction"
    current_body: List[str] = []

    heading_pattern = re.compile(r"^(#{2,6})\s+(.+?)\s*$")

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            if current_body and "\n".join(current_body).strip():
                sections.append((current_heading, current_body))
            current_heading = match.group(2).strip()
            current_body = [line]
        else:
            current_body.append(line)

    if current_body and "\n".join(current_body).strip():
        sections.append((current_heading, current_body))

    cleaned_sections: List[tuple[str, str]] = []
    for heading, body_lines in sections:
        body_text = "\n".join(body_lines).strip()
        if body_text:
            cleaned_sections.append((heading, body_text))

    return cleaned_sections


def is_table_line(line: str) -> bool:
    line = line.strip()
    return line.startswith("|") and line.endswith("|")


def is_table_separator_line(line: str) -> bool:
    line = line.strip()
    if not (line.startswith("|") and line.endswith("|")):
        return False

    inner = line.strip("|").strip()
    if not inner:
        return False

    parts = [p.strip() for p in inner.split("|")]
    return all(re.fullmatch(r":?-{3,}:?", p) for p in parts)


def extract_blocks(section_text: str) -> List[dict]:
    """
    Extract ordered blocks from a section:
    - text
    - markdown tables
    """
    lines = section_text.splitlines()
    blocks: List[dict] = []
    i = 0

    while i < len(lines):
        if (
            i + 1 < len(lines)
            and is_table_line(lines[i])
            and is_table_separator_line(lines[i + 1])
        ):
            table_lines = [lines[i], lines[i + 1]]
            i += 2

            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1

            blocks.append({
                "type": "table",
                "text": "\n".join(table_lines).strip(),
            })
            continue

        text_lines = [lines[i]]
        i += 1

        while i < len(lines):
            if (
                i + 1 < len(lines)
                and is_table_line(lines[i])
                and is_table_separator_line(lines[i + 1])
            ):
                break
            text_lines.append(lines[i])
            i += 1

        text_block = "\n".join(text_lines).strip()
        if text_block:
            blocks.append({
                "type": "text",
                "text": text_block,
            })

    return blocks


def remove_section_heading_from_start(chunk_text: str, section_title: str) -> str:
    """
    Remove repeated markdown heading from the start of a chunk.
    Example:
      ## 1. General Limited Warranty
    """
    lines = chunk_text.splitlines()
    if not lines:
        return chunk_text.strip()

    first_line = lines[0].strip()
    cleaned_section_title = section_title.strip().lower()

    if first_line.startswith("#"):
        first_line_text = re.sub(r"^#+\s*", "", first_line).strip().lower()
        if first_line_text == cleaned_section_title:
            return "\n".join(lines[1:]).strip()

    return chunk_text.strip()


def split_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Chunk prose with paragraph preference, then line fallback, plus overlap.
    """
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= chunk_size:
            current = para
            continue

        lines = para.splitlines()
        temp = ""

        for line in lines:
            candidate = f"{temp}\n{line}".strip() if temp else line
            if len(candidate) <= chunk_size:
                temp = candidate
            else:
                if temp:
                    chunks.append(temp)
                temp = line

        if temp:
            current = temp

    if current:
        chunks.append(current)

    final_chunks: List[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            final_chunks.append(chunk)
        else:
            prev_tail = chunks[idx - 1][-overlap:].strip()
            merged = f"{prev_tail}\n\n{chunk}".strip()
            final_chunks.append(merged)

    return final_chunks


def split_table_by_rows(table_text: str, max_chars: int = 1200) -> List[str]:
    """
    Keep table whole if possible.
    Otherwise split by rows and repeat header in each chunk.
    """
    lines = [line for line in table_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return [table_text.strip()]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    full_table = "\n".join(lines).strip()
    if len(full_table) <= max_chars:
        return [full_table]

    chunks: List[str] = []
    current_rows: List[str] = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate_table = "\n".join([header, separator] + candidate_rows).strip()

        if len(candidate_table) <= max_chars:
            current_rows = candidate_rows
        else:
            if current_rows:
                chunks.append("\n".join([header, separator] + current_rows).strip())
            current_rows = [row]

    if current_rows:
        chunks.append("\n".join([header, separator] + current_rows).strip())

    return chunks


def create_policy_chunks(
    file_path: str,
    text_chunk_size: int = 500,
    text_overlap: int = 100,
    table_max_chars: int = 1200,
) -> List[PolicyChunk]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    markdown_text = normalize_text(path.read_text(encoding="utf-8"))
    policy_title = extract_policy_title(markdown_text, file_path=file_path)
    sections = split_markdown_into_sections(markdown_text)

    final_chunks: List[PolicyChunk] = []

    for section_heading, section_text in sections:
        blocks = extract_blocks(section_text)
        chunk_index_in_section = 1

        for block in blocks:
            if block["type"] == "text":
                text_chunks = split_text_with_overlap(
                    text=block["text"],
                    chunk_size=text_chunk_size,
                    overlap=text_overlap,
                )

                for idx, chunk_text in enumerate(text_chunks, start=1):
                    chunk_text = normalize_text(chunk_text)

                    if idx == 1:
                        chunk_text = remove_section_heading_from_start(chunk_text, section_heading)

                    chunk_id = (
                        f"{policy_title.lower()}::{section_heading.lower()}::{chunk_index_in_section}"
                    )
                    chunk_id = re.sub(r"\s+", "_", chunk_id)
                    chunk_id = re.sub(r"[^a-z0-9_:.-]", "", chunk_id)

                    final_chunks.append(
                        PolicyChunk(
                            policy_title=policy_title,
                            section_title=section_heading,
                            text=section_heading + ":  " + chunk_text,
                            chunk_id=chunk_id,
                            chunk_index_in_section=chunk_index_in_section,
                            block_type="text",
                        )
                    )
                    chunk_index_in_section += 1

            elif block["type"] == "table":
                table_chunks = split_table_by_rows(
                    table_text=block["text"],
                    max_chars=table_max_chars,
                )

                for table_chunk in table_chunks:
                    table_chunk = normalize_text(table_chunk)

                    chunk_id = (
                        f"{policy_title.lower()}::{section_heading.lower()}::{chunk_index_in_section}"
                    )
                    chunk_id = re.sub(r"\s+", "_", chunk_id)
                    chunk_id = re.sub(r"[^a-z0-9_:.-]", "", chunk_id)

                    final_chunks.append(
                        PolicyChunk(
                            policy_title=policy_title,
                            section_title=section_heading,
                            text=section_heading + ":  " + table_chunk,
                            chunk_id=chunk_id,
                            chunk_index_in_section=chunk_index_in_section,
                            block_type="table",
                        )
                    )
                    chunk_index_in_section += 1

    return final_chunks


if __name__ == "__main__":
    file_path = "data/sample-policy.md"

    chunks = create_policy_chunks(
        file_path=file_path,
        text_chunk_size=450,
        text_overlap=75,
        table_max_chars=1600,
    )

    print(f"Total chunks: {len(chunks)}\n")

    for chunk in chunks:
        print(asdict(chunk))
        print()