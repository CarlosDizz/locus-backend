export interface AdminUser {
  id: string;
  email: string;
  display_name: string;
  avatar_url: string | null;
  roles: string[];
}

export interface AdminAuthConfig {
  google_client_id: string | null;
  local_login_enabled: boolean;
}
