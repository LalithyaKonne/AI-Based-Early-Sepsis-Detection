from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import csv
from pathlib import Path

router = APIRouter()

# Simple user store – CSV with username,password columns (no hashing for demo)
USERS_CSV = Path(__file__).resolve().parents[3] / "data" / "users.csv"

def _load_users():
    users = {}
    if USERS_CSV.is_file():
        with open(USERS_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                users[row["username"];] = row["password"]
    return users

@router.post("/login")
async def login(payload: dict):
    """Expect JSON with 'username' and 'password'. Returns a dummy token on success."""
    username = payload.get("username")
    password = payload.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing username or password")
    users = _load_users()
    stored_pw = users.get(username)
    if stored_pw is None or stored_pw != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # For simplicity, return a static token; in production replace with JWT
    token = "dummy-token-for-" + username
    return JSONResponse(content={"access_token": token, "token_type": "bearer"})
