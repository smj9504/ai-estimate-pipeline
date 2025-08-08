"""
MCP (Model Context Protocol) Client Implementation for AI Estimate Pipeline
Enables integration with MCP servers for browser automation, data fetching, etc.
"""

import json
import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """MCP JSON-RPC 2.0 Request structure"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = None
    id: Union[str, int] = 1


@dataclass
class MCPResponse:
    """MCP JSON-RPC 2.0 Response structure"""
    jsonrpc: str = "2.0"
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    id: Union[str, int] = 1


class MCPClient:
    """Base MCP Client for communicating with MCP servers"""
    
    def __init__(self, server_name: str, command: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize MCP Client
        
        Args:
            server_name: Name of the MCP server (for logging)
            command: Command to start the MCP server
            env: Environment variables for the server
        """
        self.server_name = server_name
        self.command = command
        self.env = env or os.environ.copy()
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server process"""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                bufsize=1
            )
            logger.info(f"Started MCP server: {self.server_name}")
            
            # Initialize the connection
            await self._initialize()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server {self.server_name}: {e}")
            raise
            
    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
            logger.info(f"Stopped MCP server: {self.server_name}")
            
    async def _initialize(self):
        """Initialize the MCP connection"""
        # Send initialization request
        response = await self.send_request("initialize", {
            "protocolVersion": "1.0.0",
            "capabilities": {
                "tools": True,
                "resources": True
            }
        })
        
        if response.error:
            raise Exception(f"Failed to initialize: {response.error}")
            
        # Send initialized notification
        await self.send_notification("initialized", {})
        
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> MCPResponse:
        """
        Send a request to the MCP server and wait for response
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
            
        Returns:
            MCPResponse object
        """
        self.request_id += 1
        request = MCPRequest(
            method=method,
            params=params or {},
            id=self.request_id
        )
        
        # Send request
        request_json = json.dumps({
            "jsonrpc": request.jsonrpc,
            "method": request.method,
            "params": request.params,
            "id": request.id
        })
        
        self.process.stdin.write(request_json + "\n")
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline()
        if response_line:
            response_data = json.loads(response_line)
            return MCPResponse(**response_data)
        else:
            return MCPResponse(error={"message": "No response from server"})
            
    async def send_notification(self, method: str, params: Dict[str, Any] = None):
        """
        Send a notification (no response expected)
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        notification_json = json.dumps(notification)
        self.process.stdin.write(notification_json + "\n")
        self.process.stdin.flush()
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        response = await self.send_request("tools/list")
        if response.result:
            return response.result.get("tools", [])
        return []
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """
        Call a specific tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        response = await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {}
        })
        
        if response.error:
            raise Exception(f"Tool call failed: {response.error}")
            
        return response.result


class BrowserbaseMCPClient(MCPClient):
    """Specialized client for Browserbase MCP server"""
    
    def __init__(self):
        command = ["npx", "@browserbasehq/mcp-server-browserbase"]
        env = os.environ.copy()
        env.update({
            "BROWSERBASE_API_KEY": os.getenv("BROWSERBASE_API_KEY", ""),
            "BROWSERBASE_PROJECT_ID": os.getenv("BROWSERBASE_PROJECT_ID", ""),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")
        })
        super().__init__("browserbase", command, env)
        
    async def create_session(self) -> str:
        """Create a new browser session"""
        result = await self.call_tool("browserbase_create_session")
        return result.get("sessionId")
        
    async def navigate(self, session_id: str, url: str):
        """Navigate to a URL in the browser session"""
        return await self.call_tool("browserbase_navigate", {
            "sessionId": session_id,
            "url": url
        })
        
    async def extract_text(self, session_id: str, selector: str = "body"):
        """Extract text from the current page"""
        return await self.call_tool("browserbase_extract_text", {
            "sessionId": session_id,
            "selector": selector
        })


class PlaywrightMCPClient(MCPClient):
    """Specialized client for Playwright MCP server"""
    
    def __init__(self):
        command = ["npx", "-y", "@automatalabs/mcp-server-playwright"]
        super().__init__("playwright", command)
        
    async def screenshot(self, url: str, path: str):
        """Take a screenshot of a webpage"""
        return await self.call_tool("playwright_screenshot", {
            "url": url,
            "path": path
        })
        
    async def scrape(self, url: str, selector: str):
        """Scrape content from a webpage"""
        return await self.call_tool("playwright_scrape", {
            "url": url,
            "selector": selector
        })


class ConstructionMCPClient:
    """
    High-level MCP client for construction estimate pipeline
    Coordinates multiple MCP servers for different tasks
    """
    
    def __init__(self):
        self.browserbase: Optional[BrowserbaseMCPClient] = None
        self.playwright: Optional[PlaywrightMCPClient] = None
        self.active_clients: List[MCPClient] = []
        
    @asynccontextmanager
    async def session(self):
        """Context manager for MCP session"""
        try:
            # Start required MCP servers
            await self.start_servers()
            yield self
        finally:
            # Clean up
            await self.stop_servers()
            
    async def start_servers(self):
        """Start all configured MCP servers"""
        # Start Browserbase if API key is available
        if os.getenv("BROWSERBASE_API_KEY"):
            self.browserbase = BrowserbaseMCPClient()
            await self.browserbase.start()
            self.active_clients.append(self.browserbase)
            
        # Start Playwright
        self.playwright = PlaywrightMCPClient()
        await self.playwright.start()
        self.active_clients.append(self.playwright)
        
    async def stop_servers(self):
        """Stop all active MCP servers"""
        for client in self.active_clients:
            await client.stop()
        self.active_clients.clear()
        
    async def fetch_market_prices(self, materials: List[str]) -> Dict[str, float]:
        """
        Fetch market prices for construction materials
        Uses MCP servers to scrape pricing data from various sources
        
        Args:
            materials: List of material names
            
        Returns:
            Dictionary mapping material names to prices
        """
        prices = {}
        
        # Example: Use Playwright to scrape Home Depot prices
        if self.playwright:
            for material in materials:
                try:
                    # Construct search URL
                    search_url = f"https://www.homedepot.com/s/{material.replace(' ', '%20')}"
                    
                    # Scrape price data
                    result = await self.playwright.scrape(
                        search_url,
                        ".price-format__main-price"
                    )
                    
                    if result:
                        # Parse price from result
                        price_text = result.get("text", "0")
                        price = float(price_text.replace("$", "").replace(",", ""))
                        prices[material] = price
                        
                except Exception as e:
                    logger.error(f"Failed to fetch price for {material}: {e}")
                    prices[material] = 0.0
                    
        return prices
        
    async def validate_contractor_license(self, license_number: str, state: str = "VA") -> bool:
        """
        Validate contractor license using browser automation
        
        Args:
            license_number: Contractor license number
            state: State code (default: VA for Virginia)
            
        Returns:
            True if license is valid
        """
        if self.browserbase:
            try:
                # Create browser session
                session_id = await self.browserbase.create_session()
                
                # Navigate to state licensing board
                url = f"https://www.dpor.virginia.gov/LicenseLookup"
                await self.browserbase.navigate(session_id, url)
                
                # Extract and validate license info
                result = await self.browserbase.extract_text(session_id)
                
                # Check if license is valid (simplified logic)
                return license_number in result.get("text", "")
                
            except Exception as e:
                logger.error(f"License validation failed: {e}")
                return False
                
        return False


# Example usage for testing
async def main():
    """Example usage of MCP client"""
    
    # Create construction MCP client
    client = ConstructionMCPClient()
    
    async with client.session():
        # Fetch market prices
        materials = ["2x4 lumber", "drywall", "paint"]
        prices = await client.fetch_market_prices(materials)
        print(f"Market prices: {prices}")
        
        # Validate contractor license
        is_valid = await client.validate_contractor_license("123456789")
        print(f"License valid: {is_valid}")


if __name__ == "__main__":
    asyncio.run(main())