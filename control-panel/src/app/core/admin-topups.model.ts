export interface TopUpTotals {
  top_ups: number;
  paying_users: number;
  gross_cents: number;
  estimated_fees_cents: number;
  net_cents: number;
  average_cents: number;
  bonus_cents: number;
  pending_top_ups: number;
  pending_cents: number;
  /** Saldo sin gastar de todos los monederos. Es de hoy, no del periodo. */
  outstanding_balance_cents: number;
}

export interface TopUpDailyPoint {
  day: string;
  top_ups: number;
  gross_cents: number;
}

export interface TopUpProviderBreakdown {
  provider: string;
  top_ups: number;
  gross_cents: number;
  estimated_fees_cents: number;
}

export interface TopUpItem {
  id: string;
  user_email: string;
  user_name: string;
  amount_cents: number;
  bonus_cents: number;
  estimated_fee_cents: number;
  provider: string;
  provider_reference: string;
  status: string;
  created_at: string;
  completed_at: string | null;
}

export interface AdminTopUpsDashboard {
  period_days: number;
  totals: TopUpTotals;
  daily: TopUpDailyPoint[];
  by_provider: TopUpProviderBreakdown[];
  recent: TopUpItem[];
}
