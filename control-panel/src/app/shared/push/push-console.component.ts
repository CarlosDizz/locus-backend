import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

export interface PushRecipient {
  id: number;
  display_name: string;
  email: string;
  devices: number;
  apps: string[];
  last_seen_at: string | null;
}

export interface PushResult {
  user_id: number;
  display_name: string;
  device: string;
  status: string;
  detail: string;
}

const STATUS_LABELS: Record<string, string> = {
  enviada: 'Enviada',
  simulada: 'Simulada (sin clave)',
  fallida: 'Fallida',
  caducado: 'Móvil caducado',
  sin_movil: 'Sin móvil',
  android: 'Android (sin FCM)',
};

/**
 * Avisos de prueba (portado de BACUL): elegir a alguien con móvil registrado y
 * mandarle un aviso al momento, para comprobar la cadena entera sin esperar a que
 * nadie cambie un viaje compartido.
 */
@Component({
  selector: 'locus-push-console',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './push-console.component.html',
  styleUrl: '../reviews/reviews-console.component.scss',
})
export class PushConsoleComponent {
  private readonly http = inject(HttpClient);

  readonly recipients = signal<PushRecipient[]>([]);
  readonly selected = signal<Set<number>>(new Set());
  readonly loading = signal(true);
  readonly sending = signal(false);
  readonly error = signal<string | null>(null);
  readonly results = signal<PushResult[]>([]);
  readonly apns = signal<string | null>(null);

  title = 'Borea';
  body = 'Aviso de prueba desde el panel.';
  tripId = '';

  constructor() {
    this.load();
  }

  load(): void {
    this.loading.set(true);
    this.error.set(null);
    this.http.get<PushRecipient[]>('/admin/v2/avisos/destinatarios').subscribe({
      next: (rows) => {
        this.recipients.set(rows);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No se pudieron leer los destinatarios.');
        this.loading.set(false);
      },
    });
  }

  toggle(id: number): void {
    const next = new Set(this.selected());
    if (next.has(id)) next.delete(id);
    else next.add(id);
    this.selected.set(next);
  }

  label(status: string): string {
    return STATUS_LABELS[status] ?? status;
  }

  send(): void {
    if (!this.selected().size || this.sending()) return;
    this.sending.set(true);
    this.results.set([]);
    this.http
      .post<{ apns: string; results: PushResult[] }>('/admin/v2/avisos/prueba', {
        user_ids: [...this.selected()],
        title: this.title.trim(),
        body: this.body.trim(),
        trip_id: this.tripId.trim(),
      })
      .subscribe({
        next: (response) => {
          this.apns.set(response.apns);
          this.results.set(response.results);
          this.sending.set(false);
        },
        error: () => {
          this.error.set('No se pudo enviar el aviso.');
          this.sending.set(false);
        },
      });
  }
}
