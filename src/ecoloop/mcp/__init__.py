"""Model Context Protocol server exposing the building to any MCP client.

Deliberately empty of imports: ``python -m ecoloop.mcp.server`` would otherwise
import ``server`` twice (once through this package, once as ``__main__``), which
makes ``runpy`` warn about unpredictable behaviour.
"""
