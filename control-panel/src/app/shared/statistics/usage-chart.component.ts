import { CurrencyPipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';
import { NgxEchartsDirective } from 'ngx-echarts';

import { DailyUsageSummary } from '../../core/admin-overview.model';

@Component({
  selector: 'locus-usage-chart',
  standalone: true,
  imports: [CurrencyPipe, NgxEchartsDirective],
  template: `
    <article class="chart-shell">
      <header>
        <div><p>ÚLTIMOS 14 DÍAS</p><h2>Coste y cobro</h2></div>
        <div class="totals">
          <span><i class="charged"></i>{{ chargedTotal() / 100 | currency:'EUR' }}</span>
          <span><i class="cost"></i>{{ costTotal() / 100 | currency:'EUR' }}</span>
        </div>
      </header>
      <div echarts [options]="options()" class="chart" aria-label="Evolución diaria de costes y cobros"></div>
      @if (!hasActivity()) {
        <p class="empty">Todavía no hay consumo. La gráfica empezará a dibujarse con la primera interacción.</p>
      }
    </article>
  `,
  styles: [`
    :host { display: block; }
    .chart-shell { min-height: 350px; padding: 28px; background: rgba(255,250,242,.72); border: 1px solid var(--locus-line); box-shadow: 0 16px 50px rgba(44,49,46,.055); }
    header { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; }
    header p { color: var(--locus-blue); font-size: 9px; letter-spacing: .24em; font-weight: 800; margin: 0 0 9px; }
    h2 { font: 600 25px/1.1 "Fraunces", serif; margin: 0; }
    .totals { display: flex; gap: 16px; color: var(--locus-muted); font-size: 10px; font-weight: 700; }
    .totals span { display: flex; align-items: center; gap: 6px; }
    .totals i { width: 8px; height: 8px; border-radius: 50%; }
    .charged { background: var(--locus-terracotta); }.cost { background: var(--locus-blue); }
    .chart { height: 250px; margin-top: 14px; }
    .empty { margin: -20px 0 0; text-align: center; color: var(--locus-muted); font-size: 10px; }
  `],
})
export class UsageChartComponent {
  readonly data = input.required<DailyUsageSummary[]>();
  readonly chargedTotal = computed(() => this.data().reduce((sum, row) => sum + row.charged_cents, 0));
  readonly costTotal = computed(() => this.data().reduce((sum, row) => sum + row.provider_cost_eur_cents, 0));
  readonly hasActivity = computed(() => this.data().some((row) => row.interactions > 0));
  readonly options = computed<EChartsCoreOption>(() => ({
    animationDuration: 650,
    color: ['#b86a4b', '#2f5d62'],
    grid: { left: 12, right: 12, top: 25, bottom: 10, containLabel: true },
    tooltip: { trigger: 'axis', valueFormatter: (value: unknown) => `${(Number(value) / 100).toFixed(2)} €` },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: this.data().map((row) => row.day.slice(5)),
      axisLine: { lineStyle: { color: 'rgba(47,93,98,.18)' } },
      axisLabel: { color: '#697477', fontSize: 9 },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#697477', fontSize: 9, formatter: (value: number) => `${(value / 100).toFixed(2)} €` },
      splitLine: { lineStyle: { color: 'rgba(47,93,98,.09)' } },
    },
    series: [
      { name: 'Cobrado', type: 'line', smooth: true, symbol: 'none', areaStyle: { opacity: .1 }, data: this.data().map((row) => row.charged_cents) },
      { name: 'Coste proveedor', type: 'line', smooth: true, symbol: 'none', data: this.data().map((row) => row.provider_cost_eur_cents) },
    ],
  }));
}
