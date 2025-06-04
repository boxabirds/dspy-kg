from landscape import crawl_pages_with_firecrawl, build_landscape_knowledge_graph

# Test with a single URL
print("Testing with a single URL...")
urls = ["https://jasper.ai"]

# Crawl the page
pages = crawl_pages_with_firecrawl(urls)

if pages:
    print(f"\nSuccessfully crawled {len(pages)} page(s)")
    print(f"First 500 chars of content: {pages[0]['text'][:500]}...")
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    result = build_landscape_knowledge_graph(
        domain="AI Writing Tools",
        pages=pages
    )
    
    print("\nKnowledge Graph:")
    print(result["canonical_graph"])
    print(f"\nStats: {result['stats']}")
else:
    print("No pages were crawled successfully")