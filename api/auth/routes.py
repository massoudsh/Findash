from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any

from ..database.session import get_db
from . import security, jwt
from .schemas import Token

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

@router.post("/token", response_model=Token)
def login_for_access_token(
    db: Session = Depends(get_db), 
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = security.authenticate_user(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = jwt.create_access_token(
        data={"sub": str(user.id)}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
    } 