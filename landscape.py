import dspy
from typing import List, Dict, Optional
from rdflib import Graph
from firecrawl import FirecrawlApp
import os

# Configure DSPy with your preferred LM
# For example: dspy.configure(lm=dspy.OpenAI(model="gpt-4"))
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Initialize Firecrawl
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

class IncrementalRDFMerge(dspy.Signature):
    """Merge new RDF content into an existing canonical graph, handling alignment and conflicts."""
    canonical_rdf = dspy.InputField(desc="Current canonical RDF graph in Turtle format (empty string if first page)")
    new_page_rdf = dspy.InputField(desc="RDF extracted from a new webpage in Turtle format")
    source_url = dspy.InputField(desc="URL of the new page for provenance")
    domain = dspy.InputField(desc="Domain context to guide alignment decisions")
    merged_rdf = dspy.OutputField(desc="""Updated canonical RDF in Turtle format that:
    - Integrates new information from the page
    - Identifies and merges equivalent entities (using owl:sameAs)
    - Resolves naming variations to canonical forms
    - Preserves provenance with prov:wasDerivedFrom
    - Resolves conflicts by keeping the most specific/detailed information
    - Maintains a clean ontology structure""")

class InitialSchemaInference(dspy.Signature):
    """Infer initial schema/ontology from the first few pages."""
    domain = dspy.InputField(desc="The domain/landscape being analyzed")
    sample_pages = dspy.InputField(desc="RDF from first few pages to identify patterns")
    schema_rdf = dspy.OutputField(desc="""RDF Schema/OWL ontology in Turtle format with:
    - Common classes and their hierarchy
    - Properties with domains and ranges  
    - Canonical naming conventions
    - Common equivalence patterns for the domain""")

class IncrementalLandscapeBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.schema_inferrer = dspy.ChainOfThought(InitialSchemaInference)
        self.merger = dspy.ChainOfThought(IncrementalRDFMerge)
        
    def forward(self, domain: str, pages: List[Dict[str, str]]):
        """
        Build a unified knowledge graph by incrementally merging pages.
        
        Args:
            domain: The domain/landscape name
            pages: List of dicts with 'url', 'text', and optionally 'metadata' keys
            
        Returns:
            Canonical knowledge graph with all pages merged
        """
        canonical_rdf = ""
        
        # For larger landscapes, infer schema from first few pages
        if len(pages) > 3:
            # First, extract RDF from initial pages
            initial_rdfs = []
            for page in pages[:3]:
                page_rdf = self._extract_page_rdf(
                    domain, 
                    page['text'], 
                    page['url'],
                    page.get('metadata', {})
                )
                initial_rdfs.append(page_rdf)
            
            # Infer schema
            schema_result = self.schema_inferrer(
                domain=domain,
                sample_pages="\n---\n".join(initial_rdfs)
            )
            canonical_rdf = schema_result.schema_rdf
        
        # Incrementally merge each page
        for i, page in enumerate(pages):
            # Extract RDF from page (including metadata if available)
            page_rdf = self._extract_page_rdf(
                domain, 
                page['text'], 
                page['url'],
                page.get('metadata', {})
            )
            
            # Merge into canonical graph
            merge_result = self.merger(
                canonical_rdf=canonical_rdf,
                new_page_rdf=page_rdf,
                source_url=page['url'],
                domain=domain
            )
            
            canonical_rdf = merge_result.merged_rdf
            
        return dspy.Prediction(
            canonical_graph=canonical_rdf,
            pages_processed=len(pages)
        )
    
    def _extract_page_rdf(self, domain: str, text: str, url: str, metadata: Dict = None) -> str:
        """Extract RDF from a single page including metadata."""
        # Create a comprehensive prompt that includes metadata
        metadata_str = ""
        if metadata:
            metadata_str = f"\nPage metadata: {metadata}"
        
        full_text = text + metadata_str
        
        class ExtractPageRDF(dspy.Signature):
            """Extract RDF from webpage content."""
            domain = dspy.InputField(desc="The domain/landscape being analyzed")
            text = dspy.InputField(desc="The webpage content including metadata")
            url = dspy.InputField(desc="The URL of the webpage")
            rdf = dspy.OutputField(desc="RDF in Turtle format with entities, relationships, and properties relevant to the domain. Include source URL as provenance and incorporate any relevant metadata.")
        
        extractor = dspy.ChainOfThought(ExtractPageRDF)
        result = extractor(domain=domain, text=full_text, url=url)
        return result.rdf

# Simplified usage
def build_landscape_knowledge_graph(
    domain: str,
    pages: List[Dict[str, str]],
    output_format: str = "turtle"
) -> Dict:
    """
    Build a unified knowledge graph from competitor pages using incremental merging.
    
    Args:
        domain: The landscape domain 
        pages: List of dicts with 'url' and 'text' keys
        output_format: RDF serialization format
        
    Returns:
        Canonical knowledge graph
    """
    builder = IncrementalLandscapeBuilder()
    result = builder(domain=domain, pages=pages)
    
    # Parse and validate
    g = Graph()
    try:
        g.parse(data=result.canonical_graph, format="turtle")
        if output_format != "turtle":
            output_rdf = g.serialize(format=output_format)
        else:
            output_rdf = result.canonical_graph
            
        stats = {
            "total_triples": len(g),
            "pages_processed": result.pages_processed
        }
    except Exception as e:
        output_rdf = result.canonical_graph
        stats = {"error": str(e)}
    
    return {
        "canonical_graph": output_rdf,
        "stats": stats
    }

def crawl_pages_with_firecrawl(urls: List[str]) -> List[Dict[str, str]]:
    """
    Crawl webpages using Firecrawl to get content and metadata.
    
    Args:
        urls: List of URLs to crawl
        
    Returns:
        List of dicts with 'url', 'text', and 'metadata' keys
    """
    pages = []
    
    for url in urls:
        try:
            print(f"Crawling {url}...")
            # Scrape the URL using Firecrawl
            result = firecrawl_app.scrape_url(
                url,
                formats=['markdown']  # Get markdown format for easier text processing
            )
            
            # Debug output (commented out for cleaner output)
            # print(f"Result type: {type(result)}")
            
            # Extract content and metadata
            if result:
                # Handle ScrapeResponse object
                if hasattr(result, 'markdown'):
                    content = result.markdown
                    metadata = result.metadata if hasattr(result, 'metadata') else {}
                    
                    if content:
                        pages.append({
                            'url': url,
                            'text': content,
                            'metadata': metadata
                        })
                        print(f"Successfully crawled {url}")
                    else:
                        print(f"Warning: No markdown content in result for {url}")
                elif isinstance(result, dict):
                    # Check for different possible content keys
                    content = result.get('markdown') or result.get('content') or result.get('data', {}).get('markdown')
                    metadata = result.get('metadata') or result.get('data', {}).get('metadata', {})
                    
                    if content:
                        pages.append({
                            'url': url,
                            'text': content,
                            'metadata': metadata
                        })
                        print(f"Successfully crawled {url}")
                    else:
                        print(f"Warning: No content found in result for {url}")
                        print(f"Full result: {result}")
                else:
                    print(f"Warning: Unexpected result type for {url}: {type(result)}")
                    # Try to inspect the object
                    if hasattr(result, '__dict__'):
                        print(f"Result attributes: {result.__dict__}")
            else:
                print(f"Warning: Empty result for {url}")
                
        except Exception as e:
            print(f"Error crawling {url}: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    return pages

# Even simpler: Direct accumulation approach
class SimpleAccumulator(dspy.Module):
    """Dead simple approach: just keep asking LLM to merge RDFs."""
    
    def __init__(self):
        super().__init__()
        
        class SimpleExtractRDF(dspy.Signature):
            """Extract RDF from webpage content."""
            domain = dspy.InputField()
            text = dspy.InputField()
            url = dspy.InputField()
            rdf = dspy.OutputField(desc="Extract RDF relevant to domain")
        
        class SimpleMergeRDF(dspy.Signature):
            """Merge new RDF into existing graph."""
            current_graph = dspy.InputField()
            new_content = dspy.InputField()
            domain = dspy.InputField()
            merged_graph = dspy.OutputField(desc="Merge new RDF into current graph, aligning entities and resolving conflicts intelligently")
        
        self.extractor = dspy.ChainOfThought(SimpleExtractRDF)
        self.merger = dspy.ChainOfThought(SimpleMergeRDF)
    
    def forward(self, domain: str, pages: List[Dict[str, str]]):
        graph = ""
        
        for page in pages:
            # Extract from page (including metadata if available)
            text_with_metadata = page['text']
            if 'metadata' in page and page['metadata']:
                text_with_metadata += f"\nPage metadata: {page['metadata']}"
                
            extracted = self.extractor(
                domain=domain, 
                text=text_with_metadata, 
                url=page['url']
            )
            
            # Merge
            if graph:
                merged = self.merger(
                    current_graph=graph,
                    new_content=extracted.rdf,
                    domain=domain
                )
                graph = merged.merged_graph
            else:
                graph = extracted.rdf
                
        return dspy.Prediction(final_graph=graph)

# Example usage
if __name__ == "__main__":
    # Option 1: Use Firecrawl to crawl actual webpages
    urls_to_crawl = [
        "https://grammarly.com",
        "https://jasper.ai",
        "https://copy.ai"
    ]
    
    print("Crawling webpages with Firecrawl...")
    pages = crawl_pages_with_firecrawl(urls_to_crawl)
    
    if not pages:
        print("No pages crawled successfully. Using fallback data...")
        # Fallback to static data if crawling fails
        pages = [
            {
                "url": "https://grammarly.com",
                "text": "Grammarly offers AI-powered writing assistance with features like grammar checking, tone detection, and clarity suggestions. Plans start at $12/month.",
                "metadata": {}
            },
            {
                "url": "https://jasper.ai",
                "text": "Jasper is an AI writing assistant that helps create content. It includes grammar correction, tone adjustment, and content generation. Pricing begins at $39/month.",
                "metadata": {}
            },
            {
                "url": "https://copy.ai", 
                "text": "Copy.ai provides AI writing tools including a grammar checker and tone analyzer. The pro plan costs $36 per month.",
                "metadata": {}
            }
        ]
    
    # Build graph
    print(f"Building knowledge graph from {len(pages)} pages...")
    result = build_landscape_knowledge_graph(
        domain="AI Writing Tools",
        pages=pages
    )
    
    print("Canonical Knowledge Graph:")
    print(result["canonical_graph"])
    
    # Example of what the LLM might produce after merging:
#     print("\n\nExample merged output:")
#     print("""
# @prefix : <http://example.org/ai-writing-tools/> .
# @prefix owl: <http://www.w3.org/2002/07/owl#> .
# @prefix prov: <http://www.w3.org/ns/prov#> .

# # Canonical entities (merged from all sources)
# :Grammarly a :AIWritingTool ;
#     :name "Grammarly" ;
#     :hasFeature :GrammarChecking, :ToneDetection, :ClarityEnhancement ;
#     :monthlyPrice "12"^^xsd:decimal ;
#     prov:wasDerivedFrom <https://grammarly.com> .

# :Jasper a :AIWritingTool ;
#     :name "Jasper" ;
#     :hasFeature :GrammarChecking, :ToneDetection, :ContentGeneration ;
#     :monthlyPrice "39"^^xsd:decimal ;
#     owl:sameAs :JasperAI ;  # LLM recognized these as same entity
#     prov:wasDerivedFrom <https://jasper.ai> .

# # Canonical features (LLM aligned variations)
# :GrammarChecking a :Feature ;
#     :canonicalName "Grammar Checking" ;
#     owl:sameAs :GrammarCorrection, :GrammarChecker .
    
# :ToneDetection a :Feature ;
#     :canonicalName "Tone Detection" ;
#     owl:sameAs :ToneAdjustment, :ToneAnalyzer .
#     """)