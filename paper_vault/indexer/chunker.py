from __future__ import annotations

import re

# Section header patterns (English + Chinese academic papers)
_SECTION_RE = re.compile(
    r"^(?:\d+[\.\s]+)?(?:Abstract|Introduction|Related\s?Work|Background"
    r"|Preliminaries|Problem\s+(?:Formulation|Setup|Definition|Statement)"
    r"|Method(?:ology)?|Proposed\s+(?:Method|Approach|Framework|Model|Architecture)"
    r"|System\s+(?:Overview|Design|Architecture)|Implementation"
    r"|Experiments?|(?:Experimental\s+)?(?:Evaluation|Setup|Results?)"
    r"|(?:Results?\s+and\s+)?(?:Discussion|Analysis)"
    r"|Ablation|Case\s+Study|Qualitative|Quantitative"
    r"|Conclusion|Summary|Future\s+Work|Limitations?"
    r"|Appendix|Supplementary|References|Bibliography"
    # Chinese section headers
    r"|摘要|引言|绪论|前言|背景|相关工作|预备知识|问题(?:描述|定义|形式化|陈述)"
    r"|方法论?|提出(?:方法|方案|模型|框架|架构)|系统(?:概述|设计|架构)"
    r"|实现|实验(?:评估|设置|结果)?|(?:结果\s*(?:与|和)\s*)?(?:讨论|分析)"
    r"|消融|案例(?:研究|分析)|定性|定量"
    r"|结论|总结|未来工作|局限性?|局限"
    r"|附录|补充材料|参考文献)"
    r"(?:\s|$|\.|:|。)",
    re.IGNORECASE,
)


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[dict[str, str | int]]:
    if chunk_size is None:
        from ..config import config
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        from ..config import config
        overlap = config.CHUNK_OVERLAP
    """Split text into chunks, preferring paragraph/section boundaries.

    Returns list of dicts with: text, size, section (heading name).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    current_section = ""
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2
            continue

        para_len = len(para)

        is_header = bool(_SECTION_RE.match(para))

        if is_header and current:
            chunks.append({
                "text": current.strip(),
                "size": len(current),
                "section": current_section or "",
            })
            current_section = para
            current = para + "\n\n"
        elif len(current) + para_len > chunk_size and current:
            chunks.append({
                "text": current.strip(),
                "size": len(current),
                "section": current_section or "",
            })
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + para + "\n\n"
        else:
            current += para + "\n\n"

    if current.strip():
        chunks.append({
            "text": current.strip(),
            "size": len(current),
            "section": current_section or "",
        })

    return chunks
