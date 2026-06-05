import re

_DEF_RE   = re.compile(r'\b(is defined as|refers to|means|is known as)\b', re.I)
_APP_RE   = re.compile(r'\b(because|therefore|used for|enables|allows|helps)\b', re.I)
_IMP_RE   = re.compile(r'\b(important|significant|key|critical|essential)\b', re.I)
_HEAD_RE  = re.compile(r'^[A-Z][^a-z]{2,}$|.:$')

def _classify(sentence: str) -> str:
    if _DEF_RE.search(sentence): return "definition"
    if _APP_RE.search(sentence): return "application"
    if _IMP_RE.search(sentence): return "importance"
    return "core_idea"

def generate_notes(chunks: list[str]) -> str:
    """Group chunks by heading, classify sentences, return markdown string."""
    topics: list[dict] = []
    current_topic = {"topic": "Overview", "definition": "",
                     "core_idea": "", "importance": "", "applications": []}

    for chunk in chunks:
        lines = chunk.strip().splitlines()
        for line in lines:
            s = line.strip()
            if not s: continue
            if _HEAD_RE.match(s) and len(s) < 60:
                topics.append(current_topic)
                current_topic = {"topic": s.rstrip(":"),
                                 "definition": "", "core_idea": "",
                                 "importance": "", "applications": []}
            else:
                kind = _classify(s)
                if kind == "application":
                    current_topic["applications"].append(s)
                elif not current_topic[kind]:
                    current_topic[kind] = s

    topics.append(current_topic)

    md = "# Revision Notes\n\n"
    for t in topics:
        if not any([t["definition"], t["core_idea"],
                    t["importance"], t["applications"]]):
            continue
        md = f"## {t['topic']}\n\n"
        if t["definition"]:  md = f"**Definition:** {t['definition']}\n\n"
        if t["core_idea"]:   md = f"**Core idea:** {t['core_idea']}\n\n"
        if t["importance"]:  md = f"**Why it matters:** {t['importance']}\n\n"
        if t["applications"]:
            md = "**Applications:**\n"
            for a in t["applications"]: md = f"- {a}\n"
            md = "\n"
    return md.strip()