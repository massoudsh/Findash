from pydantic import BaseModel

class Token(BaseModel):
    """
    Pydantic model for the JWT access token response.
    """
    access_token: str
    token_type: str

class TokenPayload(BaseModel):
    """
    Pydantic model for the decoded JWT payload.
    """
    sub: str | None = None 