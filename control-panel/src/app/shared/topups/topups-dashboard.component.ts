import { CommonModule } from '@angular/common';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AdminTopUpsDashboard, TopUpItem } from '../../core/admin-topups.model';

/** Nombres para humanos. Un proveedor nuevo cae al código tal cual, que es mejor
 *  que enseñar «desconocido» y perder el dato. */
const PROVIDER_LABELS: Record<string, string> = {
  paddle: 'Paddle (web)',
  google_play: 'Google Play',
  manual: 'Manual',
  stripe: 'Stripe',
};

@Component({
  selector: 'locus-topups-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './topups-dashboard.component.html',
  styleUrl: './topups-dashboard.component.scss',
})
export class TopUpsDashboardComponent {
  private readonly http = inject(HttpClient);

  readonly dashboard = signal<AdminTopUpsDashboard | null>(null);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly maxDailyGross = computed(() => Math.max(
    1,
    ...(this.dashboard()?.daily.map((point) => point.gross_cents) ?? [1]),
  ));
  readonly maxProviderGross = computed(() => Math.max(
    1,
    ...(this.dashboard()?.by_provider.map((item) => item.gross_cents) ?? [1]),
  ));
  days = 0;

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    const params = new HttpParams().set('days', this.days);
    this.http.get<AdminTopUpsDashboard>('/admin/v2/billing/topups', { params }).subscribe({
      next: (dashboard) => {
        this.dashboard.set(dashboard);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer las recargas.');
        this.loading.set(false);
      },
    });
  }

  money(cents: number): string {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 2,
    }).format(cents / 100);
  }

  feePercent(gross: number, fees: number): string {
    if (gross <= 0) return '—';
    return `${Math.round((fees / gross) * 100)}% se lo queda la pasarela`;
  }

  providerLabel(provider: string): string {
    return PROVIDER_LABELS[provider] ?? provider;
  }

  /** La inicial del proveedor, para el monograma de la izquierda. */
  monogram(provider: string): string {
    return this.providerLabel(provider).slice(0, 1).toUpperCase();
  }

  who(item: TopUpItem): string {
    return item.user_name || item.user_email || 'Cuenta borrada';
  }

  dailyHeight(gross: number): number {
    return Math.max(3, Math.round((gross / this.maxDailyGross()) * 100));
  }

  providerWidth(gross: number): number {
    return Math.max(2, Math.round((gross / this.maxProviderGross()) * 100));
  }

  shortDate(value: string): string {
    return new Intl.DateTimeFormat('es-ES', { day: 'numeric', month: 'short' }).format(new Date(value));
  }

  /** La referencia de la pasarela es larga y sólo se usa para buscarla allí.
   *  Entera no cabe; el final es lo que distingue una de otra. */
  shortReference(reference: string): string {
    return reference.length > 18 ? `…${reference.slice(-16)}` : reference;
  }
}
