import dspy
from typing import List, Dict, Optional
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from firecrawl import FirecrawlApp
import os

# Configure DSPy
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Initialize Firecrawl
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

class ExtractCommonSchema(dspy.Signature):
    """Extract a common RDF/OWL schema from multiple webpage contents."""
    domain = dspy.InputField(desc="The domain/landscape being analyzed")
    pages_content = dspy.InputField(desc="Content from multiple webpages in the domain")
    schema_rdf = dspy.OutputField(desc="""RDF/OWL schema in Turtle format defining:
    - Common classes (e.g., :AIWritingTool, :Feature, :PricingPlan, :Company)
    - Properties with domains and ranges (e.g., :hasFeature, :monthlyPrice, :offers)
    - Proper class hierarchy (no circular or nonsensical subclass relationships)
    - Required prefixes: @prefix : <http://example.org/ai-writing-tools#> .
                        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
                        @prefix owl: <http://www.w3.org/2002/07/owl#> .
    - Focus on properties that are common across all tools in the landscape
    - Do NOT include instance data, only the schema/ontology
    - Make sure classes form a sensible hierarchy (e.g., don't make Feature a subclass of PricingPlan)""")

class ExtractInstanceData(dspy.Signature):
    """Extract instance data from a webpage using a predefined schema."""
    schema_rdf = dspy.InputField(desc="The RDF/OWL schema to follow")
    domain = dspy.InputField(desc="The domain context")
    page_content = dspy.InputField(desc="The webpage content to extract from")
    page_url = dspy.InputField(desc="The URL of the webpage")
    instance_rdf = dspy.OutputField(desc="""RDF instance data in Turtle format that:
    - Strictly follows the provided schema's classes and properties
    - Creates instances of the schema classes
    - Uses the schema's properties correctly
    - Includes prov:wasDerivedFrom for provenance
    - Uses meaningful URIs for instances""")

class SchemaBasedLandscapeBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.schema_extractor = dspy.ChainOfThought(ExtractCommonSchema)
        self.instance_extractor = dspy.ChainOfThought(ExtractInstanceData)
    
    def extract_schema(self, domain: str, pages: List[Dict[str, str]]) -> str:
        """Extract a common schema from multiple pages."""
        # Combine page contents for schema extraction
        pages_content = "\n\n---PAGE---\n\n".join([
            f"URL: {p['url']}\nContent: {p['text'][:1000]}..."  # Use first 1000 chars
            for p in pages
        ])
        
        result = self.schema_extractor(
            domain=domain,
            pages_content=pages_content
        )
        
        return result.schema_rdf
    
    def extract_instances(self, schema: str, domain: str, page: Dict[str, str]) -> str:
        """Extract instance data from a single page using the schema."""
        result = self.instance_extractor(
            schema_rdf=schema,
            domain=domain,
            page_content=page['text'],
            page_url=page['url']
        )
        
        return result.instance_rdf
    
    def forward(self, domain: str, pages: List[Dict[str, str]]):
        """Build schema and instance graphs."""
        # Step 1: Extract common schema
        print("Extracting common schema from all pages...")
        schema = self.extract_schema(domain, pages)
        
        # Step 2: Extract instances for each page
        instances = {}
        for page in pages:
            print(f"Extracting instances from {page['url']}...")
            instance_rdf = self.extract_instances(schema, domain, page)
            instances[page['url']] = instance_rdf
        
        return dspy.Prediction(
            schema=schema,
            instances=instances
        )

def crawl_pages_with_firecrawl(urls: List[str]) -> List[Dict[str, str]]:
    """Crawl webpages using Firecrawl."""
    pages = []
    
    for url in urls:
        try:
            print(f"Crawling {url}...")
            result = firecrawl_app.scrape_url(url, formats=['markdown'])
            
            if result and hasattr(result, 'markdown'):
                pages.append({
                    'url': url,
                    'text': result.markdown,
                    'metadata': result.metadata if hasattr(result, 'metadata') else {}
                })
                print(f"Successfully crawled {url}")
            else:
                print(f"Warning: No content retrieved for {url}")
                
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            
    return pages

def build_landscape_schema_and_instances(
    domain: str,
    urls: List[str],
    output_dir: str = "output"
) -> Dict:
    """
    Build a common schema and individual instance graphs.
    
    Args:
        domain: The domain/landscape name
        urls: List of URLs to analyze
        output_dir: Directory to save outputs
        
    Returns:
        Dict with schema and instances
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Crawl pages
    pages = crawl_pages_with_firecrawl(urls)
    
    if not pages:
        print("No pages crawled successfully.")
        return {
            "schema": "",
            "instances": {},
            "pages_processed": 0
        }
    
    # Build schema and instances
    builder = SchemaBasedLandscapeBuilder()
    result = builder(domain=domain, pages=pages)
    
    # Save schema
    schema_path = os.path.join(output_dir, f"{domain.lower().replace(' ', '-')}-schema.ttl")
    with open(schema_path, 'w') as f:
        f.write(result.schema)
    print(f"\nSchema saved to: {schema_path}")
    
    # Save instances (without inline schema - they should reference the schema file)
    for url, instance_rdf in result.instances.items():
        # Create filename from URL
        filename = url.replace("https://", "").replace("/", "_").replace(".", "_") + ".ttl"
        instance_path = os.path.join(output_dir, filename)
        
        # Instance data with proper owl:imports to reference the schema
        # Use relative path from instance file to schema file
        schema_filename = os.path.basename(schema_path)
        instance_content = f"""@prefix : <http://example.org/ai-writing-tools#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

# This instance file imports the schema
<> owl:imports <./{schema_filename}> .

# Instance data that conforms to the imported schema
{instance_rdf}"""
        
        with open(instance_path, 'w') as f:
            f.write(instance_content)
        print(f"Instance saved to: {instance_path}")
    
    # Also create a merged file with all instances
    merged_path = os.path.join(output_dir, f"{domain.lower().replace(' ', '-')}-all-instances.ttl")
    with open(merged_path, 'w') as f:
        # Write prefixes and import statement
        f.write("@prefix : <http://example.org/ai-writing-tools#> .\n")
        f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        f.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
        f.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")
        f.write("@prefix prov: <http://www.w3.org/ns/prov#> .\n")
        f.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n")
        f.write(f"# This file imports the schema\n")
        schema_filename = os.path.basename(schema_path)
        f.write(f"<> owl:imports <./{schema_filename}> .\n")
        f.write("\n### All Instance Data ###\n")
        
        for url, instance_rdf in result.instances.items():
            f.write(f"\n### From {url} ###\n")
            # Skip prefix declarations in individual instances to avoid duplication
            lines = instance_rdf.split('\n')
            for line in lines:
                if not line.startswith('@prefix'):
                    f.write(line + '\n')
    print(f"Merged instances saved to: {merged_path}")
    
    return {
        "schema": result.schema,
        "instances": result.instances,
        "pages_processed": len(pages)
    }

# Example usage
if __name__ == "__main__":
    urls = [
        "https://grammarly.com",
        "https://jasper.ai",
        "https://copy.ai"
    ]
    
    result = build_landscape_schema_and_instances(
        domain="AI Writing Tools",
        urls=urls
    )
    
    print("\n=== COMMON SCHEMA ===")
    print(result["schema"][:1000] + "..." if len(result["schema"]) > 1000 else result["schema"])
    
    print("\n=== INSTANCE COUNTS ===")
    for url, instance_rdf in result["instances"].items():
        # Count triples (rough estimate)
        triple_count = instance_rdf.count('\n    ') + instance_rdf.count(' ;\n')
        print(f"{url}: ~{triple_count} triples")