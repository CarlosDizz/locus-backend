import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';

import { LoginComponent } from './auth/login.component';
import { AdminUser } from './core/admin-auth.model';
import { AdminOverview, ModelSummary } from './core/admin-overview.model';
import { AuditConsoleComponent } from './shared/audit/audit-console.component';
import { BillingDashboardComponent } from './shared/billing/billing-dashboard.component';
import { OperationsCalendarComponent } from './shared/calendar/operations-calendar.component';
import { CatalogExplorerComponent } from './shared/catalog/catalog-explorer.component';
import { LogConsoleComponent } from './shared/logs/log-console.component';
import { UsageChartComponent } from './shared/statistics/usage-chart.component';
import { ControlPlaneComponent } from './shared/configuration/control-plane.component';
import { UserDirectoryComponent } from './shared/users/user-directory.component';

@Component({
  selector: 'locus-root',
  standalone: true,
  imports: [AuditConsoleComponent, BillingDashboardComponent, CatalogExplorerComponent, CommonModule, ControlPlaneComponent, LoginComponent, LogConsoleComponent, OperationsCalendarComponent, UsageChartComponent, UserDirectoryComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  private readonly http = inject(HttpClient);

  readonly loading = signal(true);
  readonly authChecking = signal(true);
  readonly admin = signal<AdminUser | null>(null);
  readonly error = signal<string | null>(null);
  readonly overview = signal<AdminOverview | null>(null);
  readonly activeSection = signal('Pulso');

  readonly navigation = [
    { label: 'Pulso', glyph: 'pulse' },
    { label: 'Conversaciones', glyph: 'voice' },
    { label: 'Prompts', glyph: 'prompt' },
    { label: 'Proveedores', glyph: 'route' },
    { label: 'Ciudades y POIs', glyph: 'map' },
    { label: 'Usuarios', glyph: 'users' },
    { label: 'Consumos', glyph: 'wallet' },
    { label: 'Registros', glyph: 'logs' },
    { label: 'Auditoría', glyph: 'audit' },
  ];

  constructor() {
    this.checkSession();
  }

  checkSession(): void {
    this.http.get<AdminUser>('/admin/v2/auth/me').subscribe({
      next: (admin) => this.onAuthenticated(admin),
      error: () => this.authChecking.set(false),
    });
  }

  onAuthenticated(admin: AdminUser): void {
    this.admin.set(admin);
    this.authChecking.set(false);
    this.loadOverview();
  }

  logout(): void {
    this.http.post('/admin/v2/auth/logout', {}).subscribe({
      next: () => {
        this.admin.set(null);
        this.overview.set(null);
      },
    });
  }

  initials(): string {
    return this.admin()?.display_name.split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase() || 'CG';
  }

  selectSection(section: string): void {
    this.activeSection.set(section);
  }

  loadOverview(): void {
    this.loading.set(true);
    this.error.set(null);
    this.http.get<AdminOverview>('/admin/v2/overview').subscribe({
      next: (overview) => {
        this.overview.set(overview);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer el estado de V2. Comprueba API y migraciones.');
        this.loading.set(false);
      },
    });
  }

  /** "v0.1.0 · 74efa33 · 8 sep 16:40", trimming whatever the backend left empty. */
  buildLabel(): string {
    const build = this.overview()?.build;
    if (!build) return '';
    const parts = [`v${build.version}`, build.commit.slice(0, 7)];
    if (build.deployed_at) {
      const when = new Date(build.deployed_at);
      if (!Number.isNaN(when.getTime())) {
        parts.push(
          when.toLocaleString('es-ES', {
            day: 'numeric',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit',
          }),
        );
      }
    }
    return parts.join(' · ');
  }

  modelStatus(model: ModelSummary): string {
    if (!model.enabled) return 'En espera';
    if (model.lifecycle === 'preview') return 'Preview';
    return 'Disponible';
  }
}
