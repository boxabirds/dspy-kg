"""
knowledge_graph_rdf_dspy.py

A DSPy program that takes a textual theme and a chunk of text and produces a theme‑contextualised RDF knowledge graph (Turtle syntax).

Dependencies:
    pip install dspy-ai rdflib openai python-dotenv

Configuration:
    - Set OPENAI_API_KEY as environment variable or .env file.

Usage:
    python knowledge_graph_rdf_dspy.py --theme "Climate Change" --file document.txt
"""

import argparse
import os
import re
from typing import List

import dspy
from dspy import Structured, Predict
from rdflib import Graph, Namespace, URIRef, Literal

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# ----------------------------
# DSPy structured triple type
# ----------------------------
class Triple(Structured):
    subject: str
    predicate: str
    object: str

class TripleExtractor(Predict):
    """LLM module that extracts subject‑predicate‑object triples relevant to the theme."""
    def __init__(self):
        super().__init__(
            Triple,
            prompt="""
You are an expert ontologist. Your task is to extract factual triples (subject, predicate, object) from the given text that are relevant to the provided theme. 
Return ONLY triples in JSON list form. Do not add commentary.

Theme: {theme}

Text:
{text}
""",
        )

# ----------------------------
# Utility functions
# ----------------------------
_slug_rx = re.compile(r"[^a-zA-Z0-9]+")

def slugify(value: str) -> str:
    """Convert strings to URL‑friendly slugs."""
    value = _slug_rx.sub("_", value.strip().lower())
    return re.sub(r"_{2,}", "_", value).strip("_")

# ----------------------------
# DSPy Program
# ----------------------------
class KnowledgeGraphProgram(dspy.Module):
    """Pipeline that extracts triples then builds an RDF graph."""
    def __init__(self, base_namespace: str = "http://example.org/"):
        super().__init__()
        self.base = Namespace(base_namespace)
        self.extractor = TripleExtractor()

    def forward(self, theme: str, text: str) -> str:
        # 1. Extract triples with DSPy
        triples: List[Triple] = self.extractor(theme=theme, text=text)

        # 2. Build RDF graph
        g = Graph()
        g.bind("ex", self.base)
        relates_to = URIRef(str(self.base) + "relatesTo")

        theme_uri = URIRef(str(self.base) + slugify(theme))
        g.add((theme_uri, URIRef("http://purl.org/dc/terms/title"), Literal(theme)))

        for t in triples:
            s_uri = URIRef(str(self.base) + slugify(t.subject))
            p_uri = URIRef(str(self.base) + slugify(t.predicate))
            # Treat objects containing whitespace as literals; simple heuristic
            o_node = Literal(t.object) if " " in t.object else URIRef(str(self.base) + slugify(t.object))

            g.add((s_uri, p_uri, o_node))
            g.add((theme_uri, relates_to, s_uri))

        return g.serialize(format="turtle")

# ----------------------------
# CLI utility
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate RDF knowledge graph with DSPy.")
    parser.add_argument("--theme", required=True, help="Theme description")
    parser.add_argument("--file", required=True, help="Path to text file")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf‑8") as f:
        chunk = f.read()

    program = KnowledgeGraphProgram()
    rdf_output = program(theme=args.theme, text=chunk)
    print(rdf_output)

if __name__ == "__main__":
    main()
