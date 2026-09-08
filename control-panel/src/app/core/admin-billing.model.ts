export interface BillingTotals {
  usage_events: number;
  voice_sessions: number;
  active_users: number;
  provider_cost_eur_cents: number;
  charged_amount_cents: number;
  gross_margin_cents: number;
  pending_events: number;
  failed_events: number;
}

export interface BillingDailyPoint {
  day: string;
  usage_events: number;
  provider_cost_eur_cents: number;
  charged_amount_cents: number;
}

export interface BillingModelBreakdown {
  provider: string;
  model: string;
  usage_events: number;
  provider_cost_eur_cents: number;
  charged_amount_cents: number;
  gross_margin_cents: number;
}

export interface BillingUsageItem {
  id: string;
  user_email: string | null;
  provider: string;
  model: string;
  interaction_type: string;
  status: string;
  provider_cost_eur_cents: number;
  charged_amount_cents: number;
  gross_margin_cents: number;
  text_tokens: number;
  audio_tokens: number;
  created_at: string;
}

export interface BillingLedgerItem {
  id: string;
  user_email: string;
  kind: string;
  description: string;
  amount_cents: number;
  balance_after_cents: number;
  created_at: string;
}

export interface AdminBillingDashboard {
  period_days: number;
  totals: BillingTotals;
  daily: BillingDailyPoint[];
  by_model: BillingModelBreakdown[];
  recent_usage: BillingUsageItem[];
  recent_ledger: BillingLedgerItem[];
}
