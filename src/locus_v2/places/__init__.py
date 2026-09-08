"""Runtime geographic place lookup for the map chat.

Distinct from `catalog/`, which owns the *persistent* POI catalog and its
bootstrap pipeline. This module answers "what is around this lat/lng right
now" — the catalog rows nearby, plus live Google Places results for things
the catalog will never hold (restaurants, bars, pharmacies). Ported from
V1's `app/services/poi_service.py` + `app/clients/maps_client.py`.
"""
