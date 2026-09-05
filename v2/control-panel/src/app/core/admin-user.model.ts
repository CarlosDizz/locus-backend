export interface AdminUserSummary {
  id: number;
  email: string;
  display_name: string;
  avatar_url: string | null;
  auth_provider: string;
  locale: string;
  status: string;
  roles: string[];
  balance_cents: number;
  voice_sessions: number;
  charged_amount_cents: number;
  created_at: string;
}

export interface AdminUserList {
  items: AdminUserSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface VoiceSessionSummary {
  id: string;
  status: string;
  locale: string;
  context_type: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface LedgerEntrySummary {
  id: string;
  kind: string;
  amount_cents: number;
  balance_after_cents: number;
  description: string;
  created_at: string;
}

export interface AdminUserDetail extends AdminUserSummary {
  public_id: string;
  legacy_v1_id: number | null;
  provider_subject: string | null;
  usage_events: number;
  provider_cost_eur_cents: number;
  recent_voice_sessions: VoiceSessionSummary[];
  ledger_entries: LedgerEntrySummary[];
}
