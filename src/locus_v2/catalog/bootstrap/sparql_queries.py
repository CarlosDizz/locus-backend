"""SPARQL query builders ported verbatim from V1 `CatalogService`.

The Wikidata type list (`VALUES ?poiType`) and the query shapes themselves
are the tuned asset here — kept byte-for-byte identical to V1's queries.
"""

_POI_TYPE_VALUES = """
    wd:Q570116
    wd:Q33506
    wd:Q207694
    wd:Q16970
    wd:Q2977
    wd:Q41176
    wd:Q174782
    wd:Q4989906
    wd:Q16560
    wd:Q839954
    wd:Q12280
    wd:Q811979
    wd:Q24354
"""

_PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
"""

_ARTICLE_AND_LABEL_BLOCK = """
  OPTIONAL {
    ?article schema:about ?poi ;
             schema:isPartOf <https://es.wikipedia.org/> .
  }
  OPTIONAL {
    ?articleEn schema:about ?poi ;
               schema:isPartOf <https://en.wikipedia.org/> .
  }
  BIND(COALESCE(?article, ?articleEn) AS ?resolvedArticle)
  OPTIONAL { ?poi wikibase:sitelinks ?sitelinks . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
"""

_SELECT = (
    "SELECT ?poi ?poiLabel ?poiDescription ?coord ?poiTypeLabel ?resolvedArticle ?sitelinks "
    "WHERE {"
)


def build_city_entity_import_query(city_entity_id: str, limit: int) -> str:
    safe_limit = max(1, min(int(limit), 80))
    fetch_limit = min(max(safe_limit * 3, 80), 240)
    return f"""
{_PREFIXES}
{_SELECT}
  {{
    ?poi wdt:P131/wdt:P131* wd:{city_entity_id} .
  }}
  UNION
  {{
    ?poi wdt:P276/wdt:P131* wd:{city_entity_id} .
  }}
  ?poi wdt:P625 ?coord .
  ?poi wdt:P31/wdt:P279* ?poiType .
  VALUES ?poiType {{{_POI_TYPE_VALUES}}}
{_ARTICLE_AND_LABEL_BLOCK}
}}
ORDER BY DESC(?sitelinks)
LIMIT {fetch_limit}
"""


def build_radius_import_query(*, lat: float, lng: float, radius_km: float, limit: int) -> str:
    safe_limit = max(1, min(int(limit), 80))
    fetch_limit = min(max(safe_limit * 2, 80), 200)
    safe_radius = min(radius_km, 12.0)
    return f"""
{_PREFIXES}
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
{_SELECT}
  SERVICE wikibase:around {{
    ?poi wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({lng} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{safe_radius}" .
  }}
  ?poi wdt:P31/wdt:P279* ?poiType .
  VALUES ?poiType {{{_POI_TYPE_VALUES}}}
{_ARTICLE_AND_LABEL_BLOCK}
}}
ORDER BY DESC(?sitelinks)
LIMIT {fetch_limit}
"""
