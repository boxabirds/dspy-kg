import dspy
from rdflib import Graph
import re

dspy.settings.configure(lm=dspy.LM("openai/gpt-4.1-mini"))

# Configure DSPy with your preferred LM
# For example: dspy.configure(lm=dspy.OpenAI(model="gpt-4"))

class DirectRDFExtraction(dspy.Signature):
    """Extract a knowledge graph from text as RDF Turtle format, contextualized by theme."""
    theme = dspy.InputField(desc="The theme to contextualize extraction")
    text = dspy.InputField(desc="The text passage to analyze")
    rdf_turtle = dspy.OutputField(desc="Complete RDF graph in Turtle format with prefixes, entities as subjects with rdf:type and rdfs:label, relationships as predicates, and properties as predicates with literal values")

class ThemeBasedKnowledgeGraphExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(DirectRDFExtraction)
        
    def forward(self, theme: str, text: str):
        # Single extraction call that directly produces RDF
        result = self.extractor(theme=theme, text=text)
        
        # Parse the Turtle string into an RDFLib graph for validation and alternative formats
        g = Graph()
        try:
            g.parse(data=result.rdf_turtle, format="turtle")
        except Exception as e:
            # If parsing fails, return the raw output with error info
            return dspy.Prediction(
                rdf_turtle=result.rdf_turtle,
                graph=None,
                error=str(e)
            )
        
        return dspy.Prediction(
            rdf_turtle=result.rdf_turtle,
            graph=g
        )

class StructuredRDFExtraction(dspy.Signature):
    """Extract a knowledge graph with explicit structure requirements."""
    theme = dspy.InputField(desc="The theme to contextualize extraction")
    text = dspy.InputField(desc="The text passage to analyze")
    rdf_content = dspy.OutputField(desc="""Generate RDF in Turtle format with:
    - @prefix declarations for your namespace, rdf:, rdfs:, and others as needed
    - Theme as a subject with rdf:type :Theme
    - Entities as subjects with rdf:type and rdfs:label
    - Relationships between entities using meaningful predicates
    - Properties as predicates with literal values
    - All content contextualized to the given theme""")

# Simplified usage function
def extract_knowledge_graph(theme: str, text: str, output_format: str = "turtle"):
    """
    Extract a theme-based knowledge graph from text.
    
    Args:
        theme: The theme to contextualize the extraction
        text: The text passage to analyze
        output_format: RDF serialization format (turtle, xml, n3, etc.)
    
    Returns:
        Dictionary containing the RDF graph in requested format
    """
    # Initialize the extractor
    extractor = ThemeBasedKnowledgeGraphExtractor()
    
    # Run extraction
    result = extractor(theme=theme, text=text)
    
    # Handle different output formats
    if result.graph and output_format != "turtle":
        output_rdf = result.graph.serialize(format=output_format)
    else:
        output_rdf = result.rdf_turtle
    
    return {
        "rdf": output_rdf,
        "format": output_format,
        "graph": result.graph,
        "error": result.error if hasattr(result, 'error') else None
    }

# Alternative: Multi-format extractor
class MultiFormatRDFExtractor(dspy.Module):
    """Extract RDF in different formats based on user preference."""
    
    def __init__(self, format="turtle"):
        super().__init__()
        self.format = format
        
        if format == "turtle":
            prompt = "Generate RDF in Turtle format with @prefix declarations"
        elif format == "ntriples":
            prompt = "Generate RDF in N-Triples format with full URIs"
        elif format == "jsonld":
            prompt = "Generate RDF in JSON-LD format"
        else:
            prompt = f"Generate RDF in {format} format"
            
        self.extractor = dspy.ChainOfThought(
            f"theme, text -> rdf_output: {prompt}"
        )
    
    def forward(self, theme: str, text: str):
        result = self.extractor(theme=theme, text=text)
        return dspy.Prediction(rdf_output=result.rdf_output)

# Example usage script
if __name__ == "__main__":
    # Example theme and text
    theme = "Climate Change and Environmental Impact"
    text = """
    The Amazon rainforest, often called the lungs of the Earth, plays a crucial role in 
    regulating global climate. Scientists estimate that it absorbs approximately 2 billion 
    tons of CO2 annually. However, deforestation has accelerated in recent years, with 
    Brazil losing over 13,000 square kilometers of Amazon rainforest in 2021. This loss 
    contributes to increased greenhouse gas emissions and threatens biodiversity, as the 
    Amazon is home to roughly 10% of all species on Earth.
    """
    
    # Extract knowledge graph
    result = extract_knowledge_graph(theme, text)
    
    # print("RDF Graph (Turtle format):")
    print(result["rdf"])
    
    if result["error"]:
        print(f"\nWarning: {result['error']}")
    
    # Example of expected output format:
#     print("\n\nExample of expected RDF Turtle format:")
#     print("""
# @prefix : <http://example.org/climate-change/> .
# @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
# @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# :climate-change-and-environmental-impact rdf:type :Theme ;
#     rdfs:label "Climate Change and Environmental Impact" .

# :amazon-rainforest rdf:type :Location ;
#     rdfs:label "Amazon rainforest" ;
#     :nickname "lungs of the Earth" ;
#     :role "regulating global climate" ;
#     :absorbs "2 billion tons of CO2 annually" ;
#     :biodiversity "10% of all species on Earth" .

# :brazil rdf:type :Country ;
#     rdfs:label "Brazil" ;
#     :lost :amazon-rainforest ;
#     :deforestation-amount "13,000 square kilometers" ;
#     :deforestation-year "2021" .

# :deforestation rdf:type :EnvironmentalIssue ;
#     rdfs:label "deforestation" ;
#     :status "accelerated in recent years" ;
#     :contributes-to :greenhouse-gas-emissions ;
#     :threatens :biodiversity .
#     """)