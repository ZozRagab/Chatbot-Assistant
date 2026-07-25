from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..database import get_db
from .. import models, schemas, utils, auth

router = APIRouter(prefix="/login", tags=["Authentication"])
@router.post("/", status_code=status.HTTP_200_OK)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # username carries the email in this flow
    user = db.query(models.User).filter(models.User.email == form.username).first()
    if not user or not utils.verify_password(form.password, user.hashed_password):
        # optional header helps OAuth2 clients
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = auth.create_token({"user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}
