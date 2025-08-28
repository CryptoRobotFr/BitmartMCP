# BitmartMCP

## Config

Create .env file and fill it

Install Packages (need uv)
`uv sync`

To add in claude_desktop_config.json

```json
{
    "mcpServers": {
      "bitmart-mcp": {
        "command": "path...\\BitmartMCP\\.venv\\Scripts\\fastmcp.exe",
        "args": [
          "run",
          "path...\\BitmartMCP\\server.py:mcp"
        ]
      }
    }
  }
```
