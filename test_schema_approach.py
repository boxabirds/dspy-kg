from landscape_schema import build_landscape_schema_and_instances

# Test with fallback data (to avoid API calls)
print("Testing schema-based approach with AI Writing Tools...")

# For testing, we'll use the fallback data
result = build_landscape_schema_and_instances(
    domain="AI Writing Tools",
    urls=["https://grammarly.com", "https://jasper.ai", "https://copy.ai"]
)

print("\nDone! Check the 'output' directory for:")
print("- ai-writing-tools-schema.ttl (the common schema)")
print("- Individual instance files for each website")
print("- ai-writing-tools-all-instances.ttl (merged instances)")

# Display a sample of the schema
print("\n=== SAMPLE OF COMMON SCHEMA ===")
lines = result["schema"].split('\n')
for line in lines[:30]:  # First 30 lines
    print(line)