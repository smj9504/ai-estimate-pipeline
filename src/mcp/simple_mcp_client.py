"""
Simple MCP Client using official mcp library
Much easier than implementing from scratch!
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv

load_dotenv()


class SimpleMCPClient:
    """간단한 MCP 클라이언트 - 기존 라이브러리 사용"""
    
    async def use_browserbase(self):
        """Browserbase MCP 서버 사용 예제"""
        server_params = StdioServerParameters(
            command="npx",
            args=["@browserbasehq/mcp-server-browserbase"],
            env={
                "BROWSERBASE_API_KEY": os.getenv("BROWSERBASE_API_KEY"),
                "BROWSERBASE_PROJECT_ID": os.getenv("BROWSERBASE_PROJECT_ID"),
            }
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 초기화
                await session.initialize()
                
                # 사용 가능한 도구 목록
                tools = await session.list_tools()
                print("Available tools:", [tool.name for tool in tools.tools])
                
                # 도구 호출 예제
                if tools.tools:
                    # 세션 생성
                    result = await session.call_tool(
                        "browserbase_create_session",
                        arguments={}
                    )
                    print(f"Session created: {result}")
                    
                    # 페이지 방문
                    if result:
                        await session.call_tool(
                            "browserbase_navigate",
                            arguments={
                                "sessionId": result.get("sessionId"),
                                "url": "https://www.homedepot.com"
                            }
                        )
                        
    async def use_playwright(self):
        """Playwright MCP 서버 사용 예제"""
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@automatalabs/mcp-server-playwright"]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 초기화
                await session.initialize()
                
                # 스크린샷 찍기
                result = await session.call_tool(
                    "playwright_screenshot",
                    arguments={
                        "url": "https://example.com",
                        "path": "screenshot.png"
                    }
                )
                print(f"Screenshot saved: {result}")
                
                # 웹 스크래핑
                result = await session.call_tool(
                    "playwright_scrape", 
                    arguments={
                        "url": "https://example.com",
                        "selector": "h1"
                    }
                )
                print(f"Scraped content: {result}")


async def main():
    """메인 실행 함수"""
    client = SimpleMCPClient()
    
    # Playwright 사용
    print("Using Playwright MCP...")
    await client.use_playwright()
    
    # Browserbase 사용 (API 키가 있을 때만)
    if os.getenv("BROWSERBASE_API_KEY"):
        print("\nUsing Browserbase MCP...")
        await client.use_browserbase()


if __name__ == "__main__":
    # MCP 라이브러리 설치 필요:
    # pip install mcp
    asyncio.run(main())