export interface UserProfile {
  id: number;
  email: string;
  name: string;
  avatar_url: string | null;
  /** v5: the hand the learner signs with */
  dominant_hand?: 'right' | 'left';
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: UserProfile;
}

export interface SignupRequest {
  email: string;
  password: string;
  name: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}
