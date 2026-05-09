from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = Field(default="")


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(min_length=20)


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str
    auth_provider: str
    avatar_url: str
    is_active: bool


class AuthResponse(BaseModel):
    token: str
    user: UserResponse
