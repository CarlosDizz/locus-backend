import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, inject, input, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  AdminConfiguration,
  ConfigModel,
  PromptDefinition,
  RoutingProfile,
} from '../../core/admin-configuration.model';

@Component({
  selector: 'locus-control-plane',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './control-plane.component.html',
  styleUrl: './control-plane.component.scss',
})
export class ControlPlaneComponent {
  private readonly http = inject(HttpClient);
  readonly mode = input.required<'prompts' | 'providers'>();
  readonly configuration = signal<AdminConfiguration | null>(null);
  readonly loading = signal(true);
  readonly notice = signal<string | null>(null);
  readonly error = signal<string | null>(null);

  primaryModelId: number | null = null;
  fallbackModelId: number | null = null;
  promptVersionId: number | null = null;
  draftContent = '';

  constructor() { this.load(); }

  load(): void {
    this.loading.set(true);
    this.http.get<AdminConfiguration>('/admin/v2/configuration').subscribe({
      next: (configuration) => {
        this.configuration.set(configuration);
        const route = configuration.routing_profiles[0];
        this.primaryModelId = route?.primary_model_id ?? null;
        this.fallbackModelId = route?.fallback_model_id ?? null;
        this.promptVersionId = route?.prompt_version_id ?? null;
        this.draftContent = configuration.prompts[0]?.versions[0]?.content ?? '';
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido cargar la configuración.');
        this.loading.set(false);
      },
    });
  }

  voiceModels(): ConfigModel[] {
    return this.configuration()?.providers.flatMap((provider) => provider.models)
      .filter((model) => model.service_kind === 'voice' && model.selectable) ?? [];
  }

  publishedVersions(): { id: number; label: string }[] {
    return this.configuration()?.prompts.flatMap((prompt) => prompt.versions)
      .filter((version) => version.status === 'published')
      .map((version) => ({ id: version.id, label: `v${version.version}` })) ?? [];
  }

  toggleModel(model: ConfigModel): void {
    this.clearMessages();
    const enabled = !model.enabled;
    this.http.patch<ConfigModel>(`/admin/v2/configuration/models/${model.id}`, {
      enabled,
      selectable: enabled && model.lifecycle !== 'disabled',
    }).subscribe({
      next: () => { this.notice.set(`${model.display_name}: estado actualizado.`); this.load(); },
      error: ({ error }) => this.error.set(error?.detail ?? 'No se pudo actualizar el modelo.'),
    });
  }

  saveRouting(route: RoutingProfile): void {
    if (!this.primaryModelId || !this.promptVersionId) return;
    this.clearMessages();
    this.http.put(`/admin/v2/configuration/routing-profiles/${route.id}`, {
      primary_model_id: Number(this.primaryModelId),
      fallback_model_id: this.fallbackModelId ? Number(this.fallbackModelId) : null,
      prompt_version_id: Number(this.promptVersionId),
    }).subscribe({
      next: () => { this.notice.set('Ruta de voz publicada.'); this.load(); },
      error: ({ error }) => this.error.set(error?.detail ?? 'No se pudo publicar la ruta.'),
    });
  }

  createVersion(prompt: PromptDefinition): void {
    this.clearMessages();
    this.http.post(`/admin/v2/configuration/prompts/${prompt.id}/versions`, {
      content: this.draftContent,
    }).subscribe({
      next: () => { this.notice.set('Borrador creado como una nueva versión.'); this.load(); },
      error: ({ error }) => this.error.set(error?.detail ?? 'No se pudo crear el borrador.'),
    });
  }

  publish(versionId: number): void {
    this.clearMessages();
    this.http.post(`/admin/v2/configuration/prompt-versions/${versionId}/publish`, {}).subscribe({
      next: () => { this.notice.set('Prompt publicado.'); this.load(); },
      error: ({ error }) => this.error.set(error?.detail ?? 'No se pudo publicar el prompt.'),
    });
  }

  private clearMessages(): void { this.notice.set(null); this.error.set(null); }
}
