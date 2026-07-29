# Tools

The functions behind the agent's tool calls. See [Agent tools](../guide/tools.md) for what
each one is *for*.

## Parameter search

::: helioai.tools.rag
    options:
      members:
        - search
        - search_batch
        - search_catalogs

## Data access

::: helioai.tools.speasy_tools

## Event catalogs

::: helioai.tools.catalog_tools
    options:
      members:
        - list_catalogs
        - get_catalog
        - get_events_timeseries
        - save_catalog

## Plasma physics

::: helioai.tools.plasmapy_tools

## Sandbox

::: helioai.tools.sandbox
    options:
      members:
        - run_python

## Sandbox helpers

Available inside `run_python`, and re-emitted into exported notebooks.

::: helioai.tools.sandbox_helpers

## Recipes

::: helioai.tools.recipes

## Literature

::: helioai.tools.literature

## Registry

::: helioai.tools.registry

## MCP client

::: helioai.tools.mcp_client
    options:
      members:
        - discover_and_register
