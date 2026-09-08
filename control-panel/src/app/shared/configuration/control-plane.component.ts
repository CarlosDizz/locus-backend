import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, inject, input, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  AdminConfiguration,
  ConfigModel,
  ConfigTool,
  PromptDefinition,
  ProviderTestResult,
  RoutingProfile,
  ServiceKind,
} from '../../core/admin-configuration.model';

interface RouteSelection {
  primaryModelId: number | null;
  fallbackModelId: number | null;
  promptVersionId: number | null;
}

interface PromptRuntimeForm {
  maxOutputTokens: number;
  temperature: number;
  reasoningEffort: string;
  verbosity: string;
  turnDetection: string;
  semanticEagerness: string;
  vadThreshold: number;
  prefixPaddingMs: number;
  silenceDurationMs: number;
  createResponse: boolean;
  allowInterruptions: boolean;
  inputTranscription: string;
  openaiVoice: string;
  googleVoice: string;
  advancedJson: string;
}

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
  readonly activeService = signal<ServiceKind>('voice');
  readonly activePromptService = signal<ServiceKind>('voice');
  readonly testingModelId = signal<number | null>(null);
  readonly testResult = signal<ProviderTestResult | null>(null);
  readonly testError = signal<string | null>(null);

  routeSelections: Record<number, RouteSelection> = {};
  draftContent: Record<number, string> = {};
  draftTools: Record<number, string[]> = {};
  draftRuntime: Record<number, PromptRuntimeForm> = {};
  readonly openaiVoices = ['marin', 'cedar', 'alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse'];
  readonly googleVoices = ['Kore', 'Aoede', 'Charon', 'Fenrir', 'Puck', 'Leda', 'Orus', 'Zephyr'];

  constructor() { this.load(); }

  load(): void {
    this.loading.set(true);
    this.http.get<AdminConfiguration>('/admin/v2/configuration').subscribe({
      next: (configuration) => {
        this.configuration.set(configuration);
        this.routeSelections = Object.fromEntries(configuration.routing_profiles.map((route) => [
          route.id,
          {
            primaryModelId: route.primary_model_id,
            fallbackModelId: route.fallback_model_id,
            promptVersionId: route.prompt_version_id,
          },
        ]));
        this.draftContent = Object.fromEntries(configuration.prompts.map((prompt) => [
          prompt.id,
          prompt.versions[0]?.content ?? '',
        ]));
        this.draftTools = Object.fromEntries(configuration.prompts.map((prompt) => [
          prompt.id,
          prompt.versions[0]?.tools.map((tool) => tool.code) ?? [],
        ]));
        this.draftRuntime = Object.fromEntries(configuration.prompts.map((prompt) => [
          prompt.id,
          this.runtimeForm(prompt.versions[0]?.runtime_config ?? {}),
        ]));
        this.loading.set(false);
      },
      error: () => {
        this.error.set('No hemos podido cargar la configuración.');
        this.loading.set(false);
      },
    });
  }

  modelsFor(service: ServiceKind): ConfigModel[] {
    return this.configuration()?.providers.flatMap((provider) => provider.models)
      .filter((model) => model.service_kind === service && model.selectable) ?? [];
  }

  modelsForProvider(providerId: number, service: ServiceKind): ConfigModel[] {
    return this.configuration()?.providers.find((provider) => provider.id === providerId)?.models
      .filter((model) => model.service_kind === service) ?? [];
  }

  routesFor(service: ServiceKind): RoutingProfile[] {
    return this.configuration()?.routing_profiles.filter((route) => route.service_kind === service) ?? [];
  }

  publishedVersions(service: ServiceKind): { id: number; label: string }[] {
    return this.configuration()?.prompts.filter((prompt) => prompt.service_kind === service)
      .flatMap((prompt) => prompt.versions
        .filter((version) => version.status === 'published')
        .map((version) => ({ id: version.id, label: `${prompt.name} · v${version.version}` }))) ?? [];
  }

  selectService(service: ServiceKind): void { this.activeService.set(service); }

  selectPromptService(service: ServiceKind): void { this.activePromptService.set(service); }

  promptsFor(service: ServiceKind): PromptDefinition[] {
    return this.configuration()?.prompts.filter((prompt) => prompt.service_kind === service) ?? [];
  }

  toolsFor(service: ServiceKind): ConfigTool[] {
    return this.configuration()?.tools.filter((tool) =>
      tool.enabled && tool.service_kinds.includes(service)) ?? [];
  }

  selectPromptVersion(prompt: PromptDefinition, version: PromptDefinition['versions'][number]): void {
    this.draftContent[prompt.id] = version.content;
    this.draftTools[prompt.id] = version.tools.map((tool) => tool.code);
    this.draftRuntime[prompt.id] = this.runtimeForm(version.runtime_config);
  }

  toolSelected(promptId: number, toolCode: string): boolean {
    return this.draftTools[promptId]?.includes(toolCode) ?? false;
  }

  togglePromptTool(promptId: number, toolCode: string): void {
    const selected = new Set(this.draftTools[promptId] ?? []);
    selected.has(toolCode) ? selected.delete(toolCode) : selected.add(toolCode);
    this.draftTools[promptId] = [...selected];
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
    const selection = this.routeSelections[route.id];
    if (!selection?.primaryModelId || !selection.promptVersionId) return;
    this.clearMessages();
    this.http.put(`/admin/v2/configuration/routing-profiles/${route.id}`, {
      primary_model_id: Number(selection.primaryModelId),
      fallback_model_id: selection.fallbackModelId ? Number(selection.fallbackModelId) : null,
      prompt_version_id: Number(selection.promptVersionId),
    }).subscribe({
      next: () => { this.notice.set(`Ruta de ${route.service_kind === 'voice' ? 'voz' : 'chat'} publicada.`); this.load(); },
      error: ({ error }) => this.error.set(error?.detail ?? 'No se pudo publicar la ruta.'),
    });
  }

  createVersion(prompt: PromptDefinition): void {
    this.clearMessages();
    const runtimeConfig = this.runtimeConfig(prompt.id, prompt.service_kind);
    if (runtimeConfig === null) return;
    this.http.post(`/admin/v2/configuration/prompts/${prompt.id}/versions`, {
      content: this.draftContent[prompt.id],
      tool_codes: this.draftTools[prompt.id] ?? [],
      runtime_config: runtimeConfig,
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

  testModel(model: ConfigModel): void {
    this.testingModelId.set(model.id);
    this.testResult.set(null);
    this.testError.set(null);
    this.http.post<ProviderTestResult>(`/admin/v2/configuration/models/${model.id}/test`, {})
      .subscribe({
        next: (result) => {
          this.testResult.set(result);
          this.testingModelId.set(null);
        },
        error: ({ error }) => {
          this.testError.set(error?.detail ?? 'La prueba no ha podido completarse.');
          this.testingModelId.set(null);
        },
      });
  }

  closeTestResult(): void {
    this.testResult.set(null);
    this.testError.set(null);
  }

  money(cents: number): string {
    return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'EUR' }).format(cents / 100);
  }

  private runtimeConfig(promptId: number, service: ServiceKind): Record<string, unknown> | null {
    const form = this.draftRuntime[promptId];
    try {
      const value: unknown = JSON.parse(form.advancedJson || '{}');
      if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error();
      return {
        ...(value as Record<string, unknown>),
        max_output_tokens: Number(form.maxOutputTokens),
        temperature: Number(form.temperature),
        ...(service === 'chat' ? {
          reasoning_effort: form.reasoningEffort,
          verbosity: form.verbosity,
        } : {
          interaction_mode: form.turnDetection === 'provider_native' ? 'full_duplex' : 'turn_based',
          turn_detection: {
            type: form.turnDetection,
            eagerness: form.semanticEagerness,
            threshold: Number(form.vadThreshold),
            prefix_padding_ms: Number(form.prefixPaddingMs),
            silence_duration_ms: Number(form.silenceDurationMs),
            interrupt_response: form.allowInterruptions,
            create_response: form.createResponse,
          },
          input_audio_transcription: { model: form.inputTranscription },
          provider_overrides: {
            openai: { voice: form.openaiVoice },
            google: { voice: form.googleVoice },
          },
        }),
      };
    } catch {
      this.error.set('Las opciones avanzadas deben ser un objeto JSON válido.');
      return null;
    }
  }

  private runtimeForm(value: Record<string, unknown>): PromptRuntimeForm {
    const turn = (value['turn_detection'] ?? {}) as Record<string, unknown>;
    const transcription = (value['input_audio_transcription'] ?? {}) as Record<string, unknown>;
    const overrides = (value['provider_overrides'] ?? {}) as Record<string, Record<string, unknown>>;
    const known = new Set(['max_output_tokens', 'temperature', 'reasoning_effort', 'verbosity', 'interaction_mode', 'turn_detection', 'input_audio_transcription', 'provider_overrides']);
    const advanced = Object.fromEntries(Object.entries(value).filter(([key]) => !known.has(key)));
    return {
      maxOutputTokens: Number(value['max_output_tokens'] ?? 1200),
      temperature: Number(value['temperature'] ?? 0.8),
      reasoningEffort: String(value['reasoning_effort'] ?? 'low'),
      verbosity: String(value['verbosity'] ?? 'medium'),
      turnDetection: String(turn['type'] ?? 'semantic_vad'),
      semanticEagerness: String(turn['eagerness'] ?? 'auto'),
      vadThreshold: Number(turn['threshold'] ?? 0.5),
      prefixPaddingMs: Number(turn['prefix_padding_ms'] ?? 300),
      silenceDurationMs: Number(turn['silence_duration_ms'] ?? 500),
      createResponse: Boolean(turn['create_response'] ?? true),
      allowInterruptions: Boolean(turn['interrupt_response'] ?? true),
      inputTranscription: String(transcription['model'] ?? 'gpt-4o-mini-transcribe'),
      openaiVoice: String(overrides['openai']?.['voice'] ?? 'marin'),
      googleVoice: String(overrides['google']?.['voice'] ?? 'Kore'),
      advancedJson: JSON.stringify(advanced, null, 2),
    };
  }

  private clearMessages(): void { this.notice.set(null); this.error.set(null); }
}
