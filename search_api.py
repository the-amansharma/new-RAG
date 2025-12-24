"""
Simple API/CLI interface for hybrid search system.
"""
import json
import logging
from typing import Optional
from search.hybrid_search import HybridSearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_search_results(search_result: dict, max_results: int = 10):
    """Pretty print search results."""
    print("\n" + "=" * 80)
    print(f"🔍 Query: {search_result['query']}")
    print("=" * 80)
    
    # Print query info
    query_info = search_result.get("query_info", {})
    entities = query_info.get("entities", {})
    
    if entities.get("notification_numbers"):
        print(f"📋 Notification Numbers: {', '.join(entities['notification_numbers'])}")
    if entities.get("tax_types"):
        print(f"💰 Tax Types: {', '.join(entities['tax_types'])}")
    if entities.get("years"):
        print(f"📅 Years: {', '.join(entities['years'])}")
    
    # Print stats
    stats = search_result.get("stats", {})
    print(f"\n📊 Search Statistics:")
    print(f"   • Semantic results: {stats.get('semantic_results', 0)}")
    print(f"   • Keyword results: {stats.get('keyword_results', 0)}")
    print(f"   • Combined results: {stats.get('combined_results', 0)}")
    print(f"   • Final results: {stats.get('final_results', 0)}")
    
    # Print results
    results = search_result.get("results", [])
    print(f"\n📄 Top {min(len(results), max_results)} Results:\n")
    
    for idx, result in enumerate(results[:max_results], 1):
        print("-" * 80)
        print(f"Result #{idx}")
        print("-" * 80)
        
        # Metadata
        if result.get("notification_no"):
            print(f"📋 Notification: {result['notification_no']}")
        if result.get("tax_type"):
            print(f"💰 Tax Type: {result['tax_type']}")
        if result.get("issued_on"):
            print(f"📅 Issued On: {result['issued_on']}")
        if result.get("page_no") is not None:
            print(f"📄 Page: {result['page_no']}")
        
        # Scores
        scores = result.get("scores", {})
        print(f"\n📊 Scores:")
        print(f"   • Semantic: {scores.get('semantic', 0):.4f}")
        print(f"   • Keyword: {scores.get('keyword', 0):.4f}")
        print(f"   • Combined: {scores.get('combined', 0):.4f}")
        print(f"   • Legal: {scores.get('legal', 0):.4f}")
        print(f"   • Final: {scores.get('final', 0):.4f}")
        
        # Search types
        search_types = result.get("search_types", [])
        if search_types:
            print(f"   • Found via: {', '.join(search_types)}")
        
        # Context information
        context_count = result.get("context_chunks_count", 1)
        matched_index = result.get("matched_chunk_index")
        if context_count > 1:
            print(f"\n📚 Context: {context_count} chunks (matched chunk at position {matched_index + 1 if matched_index is not None else '?'})")
        
        # Combined chunk text (with context)
        chunk_text = result.get("chunk_text", "")
        matched_chunk_text = result.get("matched_chunk_text", "")
        
        if chunk_text:
            # Show the matched chunk text separately if available
            if matched_chunk_text and matched_chunk_text != chunk_text:
                print(f"\n🎯 Matched Chunk:\n{matched_chunk_text[:200]}..." if len(matched_chunk_text) > 200 else f"\n🎯 Matched Chunk:\n{matched_chunk_text}")
                print(f"\n📝 Full Context ({context_count} chunks):")
                preview = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                print(preview)
            else:
                preview = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                print(f"\n📝 Content Preview:\n{preview}")
        
        # File path
        if result.get("file_path"):
            print(f"\n📁 Source: {result['file_path']}")
        
        print()


def interactive_search():
    """Interactive CLI for search."""
    print("=" * 80)
    print("🔍 Indian Tax Notifications - Hybrid Search System")
    print("=" * 80)
    print("\nThis system combines semantic search with legal accuracy.")
    print("Type your query and press Enter. Type 'quit' or 'exit' to stop.\n")
    
    try:
        search_system = HybridSearch()
        print("✅ Search system initialized!\n")
    except Exception as e:
        logger.error(f"Failed to initialize search system: {e}")
        return
    
    while True:
        try:
            query = input("🔍 Enter your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            # Perform search
            print(f"\n🔍 Searching for: {query}...")
            result = search_system.search(
                query=query,
                limit=10
            )
            
            # Print results
            print_search_results(result, max_results=10)
            
            # Ask if user wants to see more details
            choice = input("Press Enter to continue, or 'save' to save results to JSON: ").strip().lower()
            if choice == "save":
                filename = input("Enter filename (default: search_results.json): ").strip() or "search_results.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Results saved to {filename}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")


def search_from_args(query: str, limit: int = 10, output_file: Optional[str] = None):
    """Search from command line arguments."""
    try:
        search_system = HybridSearch()
        result = search_system.search(query=query, limit=limit)
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to {output_file}")
        else:
            print_search_results(result, max_results=limit)
        
        return result
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        query = " ".join(sys.argv[1:])
        output_file = None
        if "--output" in sys.argv:
            idx = sys.argv.index("--output")
            output_file = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
            query = " ".join([arg for arg in sys.argv[1:] if arg != "--output" and sys.argv[sys.argv.index(arg) - 1] != "--output"])
        
        search_from_args(query, output_file=output_file)
    else:
        # Interactive mode
        interactive_search()

