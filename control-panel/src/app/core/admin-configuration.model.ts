export interface ConfigModel {
  id: number;
  external_id: string;
  display_name: string;
  service_kind: 'chat' | 'voice';
  adapter_code: string;
  lifecycle: string;
  enabled: boolean;
  selectable: boolean;
}

export interface ConfigProvider {
  id: number;
  code: string;
  name: string;
  enabled: boolean;
  models: ConfigModel[];
}

export interface PromptVersion {
  id: number;
  version: number;
  status: string;
  content: string;
  variables: Record<string, unknown>;
  tools: ConfigTool[];
  runtime_config: Record<string, unknown>;
  published_at: string | null;
}

export interface ConfigTool {
  id: number;
  code: string;
  name: string;
  description: string;
  handler_code: string;
  enabled: boolean;
  requires_approval: boolean;
  service_kinds: ServiceKind[];
  schema: Record<string, unknown>;
}

export interface PromptDefinition {
  id: number;
  code: string;
  name: string;
  description: string;
  service_kind: ServiceKind;
  versions: PromptVersion[];
}

export interface RoutingProfile {
  id: number;
  code: string;
  name: string;
  experience_code: string;
  service_kind: ServiceKind;
  environment: string;
  status: string;
  voice_mode: string;
  primary_model_id: number;
  fallback_model_id: number | null;
  prompt_version_id: number;
}

export type ServiceKind = 'chat' | 'voice';

export interface ProviderTestUsage {
  text_input_tokens: number;
  text_output_tokens: number;
  audio_input_tokens: number;
  audio_output_tokens: number;
}

export interface ProviderTestResult {
  model_id: number;
  provider_code: string;
  model: string;
  service_kind: ServiceKind;
  reply: string;
  usage: ProviderTestUsage | null;
  usage_event_id: number | null;
  billing_status: string;
  provider_cost_eur_cents: number | null;
  charged_amount_cents: number | null;
  created_at: string;
}

export interface AdminConfiguration {
  providers: ConfigProvider[];
  prompts: PromptDefinition[];
  routing_profiles: RoutingProfile[];
  tools: ConfigTool[];
}
