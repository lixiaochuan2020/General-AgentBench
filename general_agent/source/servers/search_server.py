"""
Unified Search Server

Unified Search Benchmark MCP Server, supporting BrowseComp, Mind2Web, WebVoyager.
The three benchmarks use the same search logic, only differing in evaluation methods.

Exposed tools:
- search(query): Search the web

Internal tools:
- reset_state: Reset server state
- get_answer: Get current answer
- set_answer: Set answer

Evaluation method is determined by the external evaluator based on task type.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Add deepresearch_llm_modeling to Python path
DEEPRESEARCH_PATH = Path(__file__).parent.parent.parent.parent / "benchmarks" / "deepresearch_llm_modeling"
if DEEPRESEARCH_PATH.exists():
    sys.path.insert(0, str(DEEPRESEARCH_PATH))


class SearchServer:
    """
    Unified Search Benchmark Server
    
    Provides web search functionality, supporting BrowseComp, Mind2Web, WebVoyager benchmarks.
    Only exposes one search tool externally; internal tools are for state management.
    """
    
    def __init__(self, search_engine: str = "serper"):
        """
        Initialize Search Server
        
        Args:
            search_engine: Search engine ("serper", "clueweb", "fineweb")
        """
        self.name = "search"
        self.search_engine = search_engine
        self.mcp = FastMCP(self.name)
        
        # Search statistics
        self.search_count = 0
        self.current_answer = None
        self.task_completed = False
        
        # Record sources found during search (for Mind2Web attribution evaluation)
        self.sources = []
        self.source_contents = {}
        
        # Initialize search functionality
        self._init_search_engine()
        self._register_tools()
    
    def _init_search_engine(self) -> None:
        """Initialize search engine"""
        self.search_available = False
        
        try:
            if self.search_engine == "serper":
                from retrieval import query_serper
                self.query_func = query_serper
                self.search_available = True
            elif self.search_engine == "clueweb":
                from retrieval import query_clueweb
                self.query_func = query_clueweb
                self.search_available = True
            elif self.search_engine == "fineweb":
                from retrieval import query_fineweb
                self.query_func = query_fineweb
                self.search_available = True
            else:
                print(f"Warning: Unknown search engine: {self.search_engine}")
                self.query_func = None
        except ImportError as e:
            print(f"Warning: Search engine not available: {e}")
            self.query_func = None
    
    def reset_state(self) -> None:
        """Reset server state (called before each task)"""
        self.search_count = 0
        self.current_answer = None
        self.task_completed = False
        self.sources = []
        self.source_contents = {}
    
    def get_answer(self) -> Optional[str]:
        """Get the current submitted answer"""
        return self.current_answer
    
    def set_answer(self, answer: str) -> None:
        """Set answer (called by external agent)"""
        self.current_answer = answer
        self.task_completed = True
    
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.task_completed
    
    def _extract_urls(self, content: str) -> list[str]:
        """Extract URLs from content (for Mind2Web attribution)"""
        import re
        url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+[^\s<>"\'`|(){}[\].,;:]'
        return re.findall(url_pattern, content)

    def _register_tools(self) -> None:
        """Register tools"""
        
        server_ref = self
        
        # ===== Internal tools (not exposed to Agent) =====
        @self.mcp.tool(name="reset_state")
        def reset_state() -> str:
            """[Internal] Reset the server state."""
            server_ref.reset_state()
            return json.dumps({"status": "success", "message": "State reset"})
        
        @self.mcp.tool(name="get_answer")
        def get_answer() -> str:
            """[Internal] Get the submitted answer and statistics."""
            return json.dumps({
                "status": "success",
                "answer": server_ref.current_answer,
                "completed": server_ref.task_completed,
                "search_count": server_ref.search_count,
                "sources": server_ref.sources,
                "source_contents": server_ref.source_contents
            })
        
        @self.mcp.tool(name="set_answer")
        def set_answer(answer: str) -> str:
            """[Internal] Set the answer (called by agent after parsing <answer> tag)."""
            server_ref.current_answer = answer
            server_ref.task_completed = True
            return json.dumps({
                "status": "success",
                "message": "Answer set successfully",
                "answer": answer
            })
        
        # ===== Externally exposed tools (only web_search) =====
        @self.mcp.tool(name="web_search")
        def web_search(query: str) -> str:
            """
            Search the web for information related to your query.
            
            Use this tool to find information needed to answer the question.
            The search will return relevant web page snippets.
            
            Args:
                query: The search query. Be specific and clear about what you're looking for.
            
            Returns:
                Search results containing relevant information from the web.
            """
            if not server_ref.search_available or server_ref.query_func is None:
                return json.dumps({
                    "error": "Search engine not available",
                    "message": "Please check API keys and search engine configuration"
                })
            
            try:
                # Call search engine
                if server_ref.search_engine == "serper":
                    results = server_ref.query_func(query)
                else:
                    results = server_ref.query_func(query, num_docs=1)
                
                server_ref.search_count += 1
                
                if isinstance(results, list):
                    content = "\n\n".join(results)
                else:
                    content = str(results)
                
                # Extract and record URLs (for Mind2Web attribution evaluation)
                urls = server_ref._extract_urls(content)
                for url in urls:
                    if url not in server_ref.source_contents:
                        server_ref.sources.append(url)
                        server_ref.source_contents[url] = content[:1000]
                
                return content
                
            except Exception as e:
                return json.dumps({
                    "error": "Search failed",
                    "message": str(e)
                })


def create_server(search_engine: str = "serper") -> SearchServer:
    """Create Search Server"""
    return SearchServer(search_engine=search_engine)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Unified Search MCP Server")
    parser.add_argument(
        "--search-engine",
        type=str,
        default=os.environ.get("SEARCH_ENGINE", "serper"),
        choices=["serper", "clueweb", "fineweb"],
        help="Search engine to use (default: serper)"
    )
    args = parser.parse_args()
    
    print(f"Starting Unified Search Server with engine: {args.search_engine}")
    server = SearchServer(search_engine=args.search_engine)
    
    if server.search_available:
        print(f"Search engine '{args.search_engine}' is available")
    else:
        print(f"Warning: Search engine '{args.search_engine}' is NOT available")
    
    print("\nRegistered tools:")
    print("  - search(query): Search the web for information")
    print("  - [Internal] reset_state: Reset server state")
    print("  - [Internal] get_answer: Get submitted answer")
    print("  - [Internal] set_answer: Set answer")
    print("\nSupports: BrowseComp, Mind2Web, WebVoyager")
    
    server.mcp.run()
