from hypha.connection import Hypha
import asyncio
import os
from config import Config
import jwt
from datetime import datetime, timezone, timedelta

def get_token_expiry(token):
    try:
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        exp_timestamp = decoded_token.get('exp')
        if exp_timestamp:
            expiry_time = datetime.fromtimestamp(exp_timestamp, timezone.utc)
            return expiry_time
        else:
            print("No expiration info in the token")
            return None
    except jwt.DecodeError:
        print("Failed to decode token")
        return None
    
def get_time_left_in_minutes(expiry_time):
    current_time = datetime.now(timezone.utc)
    time_left = expiry_time - current_time
    return time_left.total_seconds() / 60 
    
def is_token_expired(token, buffer_minutes=5):
    expiry_time = get_token_expiry(token)
    if expiry_time:
        time_left = get_time_left_in_minutes(expiry_time)
        if time_left <= buffer_minutes:
            print(f"Token is expired or will expire in less than {buffer_minutes} minutes")
            return True
        else:
            print(f"Token is still valid. Time left: {time_left:.2f} minutes")
            return False
    else:
        return True
    
def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:03}:{minutes:02}:{seconds:02}"

def print_token_details(token):
    print(f"TOKEN IS: {token}")
    expiration_date = get_token_expiry(token)
    print(f"Expiration date: {expiration_date}")
    expiration_time_minutes = get_time_left_in_minutes(expiration_date)
    expiration_time = timedelta(minutes=expiration_time_minutes)
    print(f"Expiration time: {format_timedelta(expiration_time)}")

def get_token():
    token = os.getenv(Config.Workspace.TOKEN_VAR_NAME, '')
    if token == '' or is_token_expired(token):
        print(f"No token found from environment variable '{Config.Workspace.TOKEN_VAR_NAME}'")
        token = asyncio.run(Hypha.retrieve_token())
        print_token_details(token)
    return token

def set_token() -> bool:
    workspace_token = get_token()
    os.environ[Config.Workspace.TOKEN_VAR_NAME] = workspace_token
    return bool(workspace_token)


