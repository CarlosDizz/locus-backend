export interface AdminLogLevelCount {
  level: string;
  count: number;
}

export interface AdminLogItem {
  id: number;
  level: string;
  service: string;
  environment: string;
  event: string;
  message: string | null;
  trace_id: string | null;
  user_id: number | null;
  voice_session_id: number | null;
  error_type: string | null;
  error_code: string | null;
  elapsed_ms: number | null;
  context: Record<string, unknown>;
  created_at: string;
}

export interface AdminLogPage {
  total: number;
  limit: number;
  offset: number;
  levels: AdminLogLevelCount[];
  services: string[];
  items: AdminLogItem[];
}
