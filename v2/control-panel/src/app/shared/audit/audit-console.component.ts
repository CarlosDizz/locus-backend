import { CommonModule } from '@angular/common';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AdminAuditItem, AdminAuditPage } from '../../core/admin-audit.model';

@Component({
  selector: 'locus-audit-console',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './audit-console.component.html',
  styleUrl: './audit-console.component.scss',
})
export class AuditConsoleComponent {
  private readonly http = inject(HttpClient);

  readonly page = signal<AdminAuditPage | null>(null);
  readonly selected = signal<AdminAuditItem | null>(null);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly total = computed(() => this.page()?.total ?? 0);

  query = '';
  action = '';
  resourceType = '';
  days = 30;

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    let params = new HttpParams().set('q', this.query.trim()).set('days', this.days);
    if (this.action) params = params.set('action', this.action);
    if (this.resourceType) params = params.set('resource_type', this.resourceType);
    this.http.get<AdminAuditPage>('/admin/v2/audit', { params }).subscribe({
      next: (page) => {
        this.page.set(page);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer la auditoría.');
        this.loading.set(false);
      },
    });
  }

  filterByAction(action: string): void {
    this.action = this.action === action ? '' : action;
    this.load();
  }

  open(item: AdminAuditItem): void {
    this.selected.set(item);
  }

  close(): void {
    this.selected.set(null);
  }

  actionLabel(action: string): string {
    const labels: Record<string, string> = {
      'model.state.changed': 'Modelo activado/desactivado',
      'prompt.version.created': 'Versión de prompt creada',
      'prompt.version.published': 'Prompt publicado',
      'routing.changed': 'Ruta publicada',
      'routing.prompt.advanced': 'Ruta y prompt publicados',
    };
    return labels[action] ?? action;
  }

  formatted(value: Record<string, unknown> | null): string {
    return value ? JSON.stringify(value, null, 2) : '—';
  }

  changedKeys(item: AdminAuditItem): string[] {
    if (!item.before || !item.after) return [];
    const keys = new Set([...Object.keys(item.before), ...Object.keys(item.after)]);
    return [...keys].filter((key) => JSON.stringify(item.before![key]) !== JSON.stringify(item.after![key]));
  }
}
