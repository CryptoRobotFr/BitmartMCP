# BitmartMCP

## Config

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
