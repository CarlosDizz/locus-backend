import { CommonModule } from '@angular/common';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  AdminBillingDashboard,
  BillingModelBreakdown,
} from '../../core/admin-billing.model';

@Component({
  selector: 'locus-billing-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './billing-dashboard.component.html',
  styleUrl: './billing-dashboard.component.scss',
})
export class BillingDashboardComponent {
  private readonly http = inject(HttpClient);

  readonly dashboard = signal<AdminBillingDashboard | null>(null);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly maxDailyCharge = computed(() => Math.max(
    1,
    ...(this.dashboard()?.daily.map((point) => point.charged_amount_cents) ?? [1]),
  ));
  readonly maxModelCharge = computed(() => Math.max(
    1,
    ...(this.dashboard()?.by_model.map((item) => item.charged_amount_cents) ?? [1]),
  ));
  days = 0;

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    const params = new HttpParams().set('days', this.days);
    this.http.get<AdminBillingDashboard>('/admin/v2/billing', { params }).subscribe({
      next: (dashboard) => {
        this.dashboard.set(dashboard);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido reconstruir los consumos.');
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

  marginPercent(cost: number, charged: number): string {
    if (charged <= 0) return '0%';
    return `${Math.round(((charged - cost) / charged) * 100)}%`;
  }

  dailyHeight(charged: number): number {
    return Math.max(3, Math.round((charged / this.maxDailyCharge()) * 100));
  }

  modelWidth(item: BillingModelBreakdown): number {
    return Math.max(2, Math.round((item.charged_amount_cents / this.maxModelCharge()) * 100));
  }

  shortDate(value: string): string {
    return new Intl.DateTimeFormat('es-ES', { day: 'numeric', month: 'short' }).format(new Date(value));
  }
}
