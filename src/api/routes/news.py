from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.core.models import NewsArticle
from src.database.postgres_connection import get_db

router = APIRouter() 