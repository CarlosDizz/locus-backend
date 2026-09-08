import { CommonModule } from '@angular/common';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AdminUserDetail, AdminUserList, AdminUserSummary } from '../../core/admin-user.model';

@Component({
  selector: 'locus-user-directory',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './user-directory.component.html',
  styleUrl: './user-directory.component.scss',
})
export class UserDirectoryComponent {
  private readonly http = inject(HttpClient);
  readonly users = signal<AdminUserSummary[]>([]);
  readonly selected = signal<AdminUserDetail | null>(null);
  readonly loading = signal(true);
  readonly detailLoading = signal(false);
  readonly error = signal<string | null>(null);
  readonly total = signal(0);
  query = '';
  status = '';

  constructor() { this.load(); }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    let params = new HttpParams().set('q', this.query.trim());
    if (this.status) params = params.set('user_status', this.status);
    this.http.get<AdminUserList>('/admin/v2/users', { params }).subscribe({
      next: (result) => {
        this.users.set(result.items);
        this.total.set(result.total);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido leer los usuarios.');
        this.loading.set(false);
      },
    });
  }

  open(user: AdminUserSummary): void {
    this.detailLoading.set(true);
    this.http.get<AdminUserDetail>(`/admin/v2/users/${user.id}`).subscribe({
      next: (detail) => {
        this.selected.set(detail);
        this.detailLoading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido abrir la ficha del usuario.');
        this.detailLoading.set(false);
      },
    });
  }

  close(): void { this.selected.set(null); }

  initials(user: AdminUserSummary): string {
    return user.display_name.split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase();
  }

  money(cents: number): string {
    return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(cents / 100);
  }
}
