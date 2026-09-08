import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, DestroyRef, inject, input, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AdminConfiguration, RoutingProfile } from '../../core/admin-configuration.model';
import { AdminPoiDetail } from '../../core/admin-catalog.model';

interface TranscriptLine {
  role: 'user' | 'assistant' | 'tool' | 'system' | 'error';
  text: string;
}

interface ServerEvent {
  type: string;
  sequence: number;
  trace_id: string;
  payload: Record<string, any>;
}

type CallStatus = 'idle' | 'connecting' | 'ready' | 'live' | 'ended' | 'error';

/**
 * Real WebSocket call test harness for a single POI, embedded in its detail
 * panel. Talks directly to the same voice.gateway.VoiceGateway (/ws/v2/live)
 * the Ionic app's call screen uses — session.start with context_type=poi and
 * the POI's public_id, then a bidirectional PCM16/text/tool-call bridge.
 *
 * Deliberately NOT the call-room/multi-participant protocol (floor.*,
 * call.snapshot): this is a single-admin prompt/provider testing tool, not a
 * port of the group-call feature.
 */
@Component({
  selector: 'locus-poi-call-test',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './poi-call-test.component.html',
  styleUrl: './poi-call-test.component.scss',
})
export class PoiCallTestComponent implements OnInit {
  readonly poi = input.required<AdminPoiDetail>();

  private readonly http = inject(HttpClient);
  private readonly destroyRef = inject(DestroyRef);

  readonly profiles = signal<RoutingProfile[]>([]);
  readonly profilesLoading = signal(true);
  readonly profilesError = signal('');
  selectedProfileCode = '';

  readonly status = signal<CallStatus>('idle');
  readonly statusText = signal('');
  readonly providerLabel = signal('');
  readonly transcript = signal<TranscriptLine[]>([]);
  readonly usage = signal<Record<string, number> | null>(null);
  readonly talking = signal(false);
  textDraft = '';

  private socket: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private micProcessor: ScriptProcessorNode | null = null;
  private silentGain: GainNode | null = null;
  private playbackCursor = 0;
  private liveUserText = '';
  private liveAssistantText = '';

  constructor() {
    this.destroyRef.onDestroy(() => this.endCall());
  }

  ngOnInit(): void {
    this.http.get<AdminConfiguration>('/admin/v2/configuration').subscribe({
      next: (config) => {
        const voiceProfiles = config.routing_profiles.filter(
          (profile) => profile.service_kind === 'voice' && profile.status === 'published',
        );
        this.profiles.set(voiceProfiles);
        this.selectedProfileCode = voiceProfiles[0]?.code ?? '';
        this.profilesLoading.set(false);
      },
      error: () => {
        this.profilesError.set('No hemos podido leer los perfiles de voz publicados.');
        this.profilesLoading.set(false);
      },
    });
  }

  isRealProvider(): boolean {
    return !this.selectedProfileCode.endsWith('.mock');
  }

  canStart(): boolean {
    return !!this.selectedProfileCode && (this.status() === 'idle' || this.status() === 'ended' || this.status() === 'error');
  }

  isLive(): boolean {
    return this.status() === 'ready' || this.status() === 'live';
  }

  async startCall(): Promise<void> {
    if (!this.canStart()) return;
    this.transcript.set([]);
    this.usage.set(null);
    this.providerLabel.set('');
    this.status.set('connecting');
    this.statusText.set('Conectando…');

    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    this.socket = new WebSocket(`${protocol}://${location.host}/ws/v2/live`);
    this.socket.binaryType = 'arraybuffer';

    this.socket.onopen = () => {
      this.send({
        type: 'session.start',
        event_id: this.eventId(),
        locale: 'es-ES',
        routing_profile: this.selectedProfileCode,
        context_type: 'poi',
        context_id: this.poi().public_id,
        audio_format: 'pcm16_24khz',
      });
    };
    this.socket.onmessage = (message) => this.onServerEvent(message);
    this.socket.onerror = () => {
      if (this.status() !== 'ended') {
        this.status.set('error');
        this.statusText.set('Error de conexión con la pasarela de voz.');
      }
    };
    this.socket.onclose = () => {
      this.stopMicCapture();
      if (this.status() !== 'error') {
        this.status.set('ended');
        this.statusText.set('Llamada finalizada.');
      }
    };
  }

  endCall(): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.send({ type: 'session.close', event_id: this.eventId() });
      this.socket.close();
    }
    this.socket = null;
    this.stopMicCapture();
    void this.releaseAudioContext();
  }

  async startTalk(): Promise<void> {
    if (!this.isLive() || this.talking()) return;
    try {
      await this.ensureMicCapture();
    } catch {
      this.pushLine('error', 'No se ha podido acceder al micrófono.');
      return;
    }
    this.talking.set(true);
    this.send({ type: 'audio.start', event_id: this.eventId(), format: 'pcm16_24khz' });
  }

  stopTalk(): void {
    if (!this.talking()) return;
    this.talking.set(false);
    this.send({ type: 'audio.commit', event_id: this.eventId() });
    this.statusText.set('Locus está respondiendo…');
  }

  sendText(): void {
    const text = this.textDraft.trim();
    if (!text || !this.isLive()) return;
    this.pushLine('user', text);
    this.send({ type: 'text.send', event_id: this.eventId(), text });
    this.textDraft = '';
    this.statusText.set('Locus está respondiendo…');
  }

  private onServerEvent(message: MessageEvent): void {
    if (typeof message.data !== 'string') return;
    let event: ServerEvent;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    const payload = event.payload || {};
    switch (event.type) {
      case 'session.ready':
        this.status.set('ready');
        this.statusText.set('Llamada abierta. Mantén pulsado para hablar o escribe.');
        this.providerLabel.set(`${payload['provider']} · ${payload['model']}`);
        this.pushLine('system', `Sesión iniciada con ${payload['provider']} (${payload['model']}).`);
        return;
      case 'transcript.user.delta':
        this.liveUserText += payload['text'] ?? '';
        return;
      case 'transcript.user.done': {
        const text = (payload['text'] as string) || this.liveUserText;
        this.liveUserText = '';
        if (text) this.pushLine('user', text);
        return;
      }
      case 'transcript.assistant.delta':
        this.liveAssistantText += payload['text'] ?? '';
        return;
      case 'transcript.assistant.done': {
        const text = (payload['text'] as string) || this.liveAssistantText;
        this.liveAssistantText = '';
        if (text) this.pushLine('assistant', text);
        this.status.set('ready');
        this.statusText.set('Mantén pulsado para hablar o escribe.');
        return;
      }
      case 'audio.delta':
        void this.playAudioChunk(payload['audio'] as string);
        this.status.set('live');
        return;
      case 'audio.done':
        return;
      case 'tool.started':
        this.pushLine('tool', `Ejecutando ${payload['name']}…`);
        return;
      case 'tool.done':
        this.pushLine('tool', `${payload['name']} completado.`);
        return;
      case 'tool.approval_required':
        // Every seeded tool has requires_approval=false today; auto-approve
        // so a future approval-gated tool doesn't stall the test harness.
        this.pushLine('tool', `${payload['name']} pide aprobación — aprobado automáticamente.`);
        this.send({
          type: 'tool.approval',
          event_id: this.eventId(),
          call_id: payload['call_id'],
          approved: true,
        });
        return;
      case 'usage.recorded':
        this.usage.set(payload as Record<string, number>);
        return;
      case 'provider.error':
        this.pushLine('error', String(payload['message'] ?? 'Error del proveedor.'));
        return;
      case 'session.error':
        this.status.set('error');
        this.statusText.set(String(payload['message'] ?? 'Error de la sesión.'));
        this.pushLine('error', this.statusText());
        return;
      case 'session.closed':
        this.status.set('ended');
        return;
      default:
        return;
    }
  }

  private pushLine(role: TranscriptLine['role'], text: string): void {
    this.transcript.update((lines) => [...lines, { role, text }]);
  }

  private send(payload: Record<string, unknown>): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }

  private eventId(): string {
    return crypto.randomUUID();
  }

  private async ensureMicCapture(): Promise<void> {
    const context = await this.ensureAudioContext();
    if (!this.micStream) {
      this.micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    }
    if (this.micSource && this.micProcessor) return;

    this.micSource = context.createMediaStreamSource(this.micStream);
    this.micProcessor = context.createScriptProcessor(4096, 1, 1);
    this.silentGain = context.createGain();
    this.silentGain.gain.value = 0;

    this.micProcessor.onaudioprocess = (audioEvent) => {
      if (!this.talking() || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
      const channel = audioEvent.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(channel.length);
      for (let index = 0; index < channel.length; index += 1) {
        const value = Math.max(-1, Math.min(1, channel[index]));
        pcm[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
      }
      this.socket.send(pcm.buffer);
    };

    this.micSource.connect(this.micProcessor);
    this.micProcessor.connect(this.silentGain);
    this.silentGain.connect(context.destination);
  }

  private stopMicCapture(): void {
    this.talking.set(false);
    this.micProcessor?.disconnect();
    this.micSource?.disconnect();
    this.silentGain?.disconnect();
    this.micProcessor = null;
    this.micSource = null;
    this.silentGain = null;
    this.micStream?.getTracks().forEach((track) => track.stop());
    this.micStream = null;
  }

  private async ensureAudioContext(): Promise<AudioContext> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate: 24000 });
    }
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
    return this.audioContext;
  }

  private async releaseAudioContext(): Promise<void> {
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
    this.playbackCursor = 0;
  }

  private async playAudioChunk(base64Audio: string): Promise<void> {
    if (!base64Audio) return;
    const context = await this.ensureAudioContext();
    const binary = atob(base64Audio);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    const pcm = new Int16Array(bytes.buffer);
    if (!pcm.length) return;
    const samples = new Float32Array(pcm.length);
    for (let index = 0; index < pcm.length; index += 1) {
      samples[index] = pcm[index] / 0x8000;
    }
    const buffer = context.createBuffer(1, samples.length, 24000);
    buffer.getChannelData(0).set(samples);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(context.currentTime + 0.02, this.playbackCursor);
    source.start(startAt);
    this.playbackCursor = startAt + buffer.duration;
  }
}
