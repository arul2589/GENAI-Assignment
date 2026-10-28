from mcp.server.fastmcp import FastMCP

# Give the server a name
mcp = FastMCP("Grocery")

@mcp.tool()
def get_price(item: str) -> str:
    """Return today's price for a grocery item."""
    prices = {"apple": "30 rupees", "banana": "10 rupees", "milk": "50 rupees"}
    return prices.get(item.lower(), "price not available")

if __name__ == "__main__":
    # Talk over stdio (standard input/output) - the simplest MCP connection
    mcp.run()
