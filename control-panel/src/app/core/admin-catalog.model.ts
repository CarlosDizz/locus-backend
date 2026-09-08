export interface AdminCitySummary {
  id: number;
  public_id: string;
  name: string;
  names: Record<string, string>;
  slug: string;
  country_code: string;
  lat: number | string | null;
  lng: number | string | null;
  source: string;
  poi_count: number;
  active_poi_count: number;
}

export interface AdminCityList {
  total: number;
  items: AdminCitySummary[];
}

export interface AdminPoiSummary {
  id: number;
  public_id: string;
  name: string;
  slug: string;
  type_code: string | null;
  type_name: string | null;
  lat: number | string | null;
  lng: number | string | null;
  source_of_truth: string;
  is_active: boolean;
}

export interface AdminPoiPage {
  total: number;
  limit: number;
  offset: number;
  items: AdminPoiSummary[];
}

export interface AdminPoiDetail extends AdminPoiSummary {
  city_id: number | null;
  city_name: string | null;
  names: Record<string, string>;
  short_description: string;
  short_descriptions: Record<string, string>;
  long_description: string;
  wikidata_id: string;
  wikipedia_title: string;
  google_place_id: string;
  metadata: Record<string, unknown>;
}

export interface PoiTypeOption {
  code: string;
  name: string;
}

export interface PoiUpdateRequest {
  name?: string | null;
  names?: Record<string, string> | null;
  short_description?: string | null;
  short_descriptions?: Record<string, string> | null;
  long_description?: string | null;
  lat?: string | null;
  lng?: string | null;
  poi_type_code?: string | null;
  is_active?: boolean | null;
  wikidata_id?: string | null;
  wikipedia_title?: string | null;
  google_place_id?: string | null;
}

export interface BootstrapPoi {
  id: number;
  public_id: string;
  name: string;
  slug: string;
  type_code: string | null;
  lat: number | null;
  lng: number | null;
  short_description: string;
  source_of_truth: string;
  created: boolean;
}

export interface BootstrapResult {
  city_id: number;
  city_public_id: string;
  city_name: string;
  city_created: boolean;
  source: string;
  imported_count: number;
  updated_count: number;
  pois: BootstrapPoi[];
}
