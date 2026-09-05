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
  published_at: string | null;
}

export interface PromptDefinition {
  id: number;
  code: string;
  name: string;
  description: string;
  versions: PromptVersion[];
}

export interface RoutingProfile {
  id: number;
  code: string;
  name: string;
  experience_code: string;
  environment: string;
  status: string;
  voice_mode: string;
  primary_model_id: number;
  fallback_model_id: number | null;
  prompt_version_id: number;
}

export interface AdminConfiguration {
  providers: ConfigProvider[];
  prompts: PromptDefinition[];
  routing_profiles: RoutingProfile[];
}
