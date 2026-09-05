export interface AdminAuditActionCount {
  action: string;
  count: number;
}

export interface AdminAuditItem {
  id: number;
  actor_user_id: number;
  actor_name: string;
  actor_email: string;
  action: string;
  resource_type: string;
  resource_id: string;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  trace_id: string;
  created_at: string;
}

export interface AdminAuditPage {
  total: number;
  limit: number;
  offset: number;
  actions: AdminAuditActionCount[];
  resource_types: string[];
  items: AdminAuditItem[];
}
