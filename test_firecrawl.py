import os
from firecrawl import FirecrawlApp

# Test Firecrawl API
api_key = os.getenv("FIRECRAWL_API_KEY")
if not api_key:
    print("Error: FIRECRAWL_API_KEY environment variable not set")
    exit(1)

print(f"Using API key: {api_key[:8]}...")
app = FirecrawlApp(api_key=api_key)

# Test with a simple URL
test_url = "https://example.com"
print(f"\nTesting Firecrawl with {test_url}...")

try:
    result = app.scrape_url(test_url, formats=['markdown'])
    print(f"Result type: {type(result)}")
    
    if hasattr(result, 'markdown'):
        print(f"Markdown content (first 200 chars): {result.markdown[:200]}...")
        print(f"Has metadata: {hasattr(result, 'metadata')}")
        if hasattr(result, 'metadata'):
            print(f"Metadata: {result.metadata}")
    else:
        print(f"Result attributes: {dir(result)}")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()