import { CommonModule } from '@angular/common';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AdminLogItem, AdminLogPage } from '../../core/admin-log.model';

@Component({
  selector: 'locus-log-console',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './log-console.component.html',
  styleUrl: './log-console.component.scss',
})
export class LogConsoleComponent {
  private readonly http = inject(HttpClient);

  readonly page = signal<AdminLogPage | null>(null);
  readonly selected = signal<AdminLogItem | null>(null);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly problems = computed(() => this.levelCount('error') + this.levelCount('critical'));

  query = '';
  level = '';
  service = '';
  days = 7;

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    let params = new HttpParams().set('q', this.query.trim()).set('days', this.days);
    if (this.level) params = params.set('level', this.level);
    if (this.service) params = params.set('service', this.service);
    this.http.get<AdminLogPage>('/admin/v2/logs', { params }).subscribe({
      next: (page) => {
        this.page.set(page);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer el cuaderno operativo.');
        this.loading.set(false);
      },
    });
  }

  filterByLevel(level: string): void {
    this.level = this.level === level ? '' : level;
    this.load();
  }

  levelCount(level: string): number {
    return this.page()?.levels.find((item) => item.level === level)?.count ?? 0;
  }

  open(item: AdminLogItem): void {
    this.selected.set(item);
  }

  close(): void {
    this.selected.set(null);
  }

  formattedContext(item: AdminLogItem): string {
    return JSON.stringify(item.context, null, 2);
  }

  hasContext(item: AdminLogItem): boolean {
    return Object.keys(item.context).length > 0;
  }
}
