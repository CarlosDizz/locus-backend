export interface OverviewMetric {
  label: string;
  value: number;
  tone: string;
}

export interface ModelSummary {
  id: number;
  provider: string;
  external_id: string;
  display_name: string;
  service_kind: string;
  adapter_code: string;
  lifecycle: string;
  enabled: boolean;
  selectable: boolean;
  capabilities: Record<string, unknown>;
}

export interface DailyUsageSummary {
  day: string;
  interactions: number;
  charged_cents: number;
  provider_cost_eur_cents: number;
}

export interface CalendarActivity {
  id: string;
  title: string;
  start: string;
  kind: string;
}

export interface PoiMapPoint {
  id: string;
  name: string;
  city: string;
  lat: number;
  lng: number;
}

export interface AdminOverview {
  environment: string;
  metrics: OverviewMetric[];
  registered_adapters: string[];
  models: ModelSummary[];
  usage: DailyUsageSummary[];
  activities: CalendarActivity[];
  poi_map: PoiMapPoint[];
}
