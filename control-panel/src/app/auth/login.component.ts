import { HttpClient } from '@angular/common/http';
import { AfterViewInit, Component, ElementRef, inject, output, signal, viewChild } from '@angular/core';

import { AdminAuthConfig, AdminUser } from '../core/admin-auth.model';

interface GoogleCredentialResponse { credential: string; }

declare global {
  interface Window {
    google?: {
      accounts: { id: {
        initialize(options: { client_id: string; callback: (value: GoogleCredentialResponse) => void }): void;
        renderButton(element: HTMLElement, options: Record<string, unknown>): void;
      }};
    };
  }
}

@Component({
  selector: 'locus-login',
  standalone: true,
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
})
export class LoginComponent implements AfterViewInit {
  private readonly http = inject(HttpClient);
  private readonly googleButton = viewChild<ElementRef<HTMLElement>>('googleButton');
  readonly authenticated = output<AdminUser>();
  readonly config = signal<AdminAuthConfig | null>(null);
  readonly busy = signal(false);
  readonly error = signal<string | null>(null);

  ngAfterViewInit(): void {
    this.http.get<AdminAuthConfig>('/admin/v2/auth/config').subscribe({
      next: (config) => {
        this.config.set(config);
        if (config.google_client_id) this.mountGoogle(config.google_client_id);
      },
      error: () => this.error.set('No podemos contactar con la puerta de acceso.'),
    });
  }

  loginLocal(): void {
    this.busy.set(true);
    this.error.set(null);
    this.http.post<AdminUser>('/admin/v2/auth/local', {}).subscribe({
      next: (user) => this.authenticated.emit(user),
      error: () => {
        this.busy.set(false);
        this.error.set('El acceso local no está disponible.');
      },
    });
  }

  private mountGoogle(clientId: string): void {
    const ready = (): void => {
      const element = this.googleButton()?.nativeElement;
      if (!element || !window.google) return;
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: ({ credential }) => this.loginGoogle(credential),
      });
      window.google.accounts.id.renderButton(element, {
        theme: 'outline', size: 'large', shape: 'pill', text: 'continue_with', width: 310,
      });
    };
    if (window.google) return ready();
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.onload = ready;
    script.onerror = () => this.error.set('Google Identity no ha podido cargar.');
    document.head.appendChild(script);
  }

  private loginGoogle(credential: string): void {
    this.busy.set(true);
    this.error.set(null);
    this.http.post<AdminUser>('/admin/v2/auth/google', { credential }).subscribe({
      next: (user) => this.authenticated.emit(user),
      error: () => {
        this.busy.set(false);
        this.error.set('Esta cuenta de Google no tiene acceso al panel.');
      },
    });
  }
}
