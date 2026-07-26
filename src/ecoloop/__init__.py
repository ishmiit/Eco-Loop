"""Eco-Loop Building Agents.

An autonomous closed-loop supervisory controller for buildings:

    EnergyPlus  --telemetry-->  MCP tool layer  -->  open-source LLM
         ^                                                |
         |------------------ set-points ------------------|

Package layout
--------------
``ecoloop.sim``      simulation engines (EnergyPlus runtime API + surrogate)
``ecoloop.agent``    LLM client, tool registry, guardrails, fallback policy
``ecoloop.mcp``      MCP server exposing the same tool registry to any client
``ecoloop.server``   FastAPI + live dashboard (SSE)
``ecoloop.cli``      ``python -m ecoloop ...``
"""

__version__ = "1.0.0"
