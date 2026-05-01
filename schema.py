from dataclasses import dataclass
from typing import Optional

@dataclass
class Paper:
    paper_id:   str          # e.g. "2401.12345"
    title:      str
    abstract:   str
    body_text:  Optional[str]  # None if extraction failed
    url:        str          # https://arxiv.org/abs/{paper_id}
    pdf_url:    str          # https://arxiv.org/pdf/{paper_id}
    year:       int          # 2023 or 2024
    category:   str          # "cs.LG" or "cs.AI"
    authors:    list[str]
    source:     str          # "full_text" | "abstract_only"