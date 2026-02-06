"""
Stage 2: AI-Powered Resume Parser
This module takes extracted resume text and uses AI to parse it into structured data.
"""
import json
import os
import time
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from typing import List, Optional
import models

MAX_REQUESTS_PER_MINUTE = 4
MAX_REQUESTS_PER_DAY = 19
SECONDS_BETWEEN_REQUESTS = 60 / MAX_REQUESTS_PER_MINUTE  # 15 seconds
QUOTA_STATE_FILE = "quota_state.json"  # File to persist quota across runs


def load_quota_state():
    """Load quota state from file"""
    if not os.path.exists(QUOTA_STATE_FILE):
        return {
            "daily_request_count": 0,
            "daily_reset_date": None,
            "request_timestamps": []
        }
    
    try:
        with open(QUOTA_STATE_FILE, 'r') as f:
            state = json.load(f)
            # Convert timestamp strings back to datetime objects
            state["request_timestamps"] = [
                datetime.fromisoformat(ts) for ts in state.get("request_timestamps", [])
            ]
            return state
    except Exception as e:
        print(f"⚠️  Error loading quota state: {e}. Starting fresh.")
        return {
            "daily_request_count": 0,
            "daily_reset_date": None,
            "request_timestamps": []
        }


def save_quota_state(daily_count, reset_date, timestamps):
    """Save quota state to file"""
    state = {
        "daily_request_count": daily_count,
        "daily_reset_date": reset_date.isoformat() if reset_date else None,
        "request_timestamps": [ts.isoformat() for ts in timestamps]
    }
    
    try:
        with open(QUOTA_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠️  Error saving quota state: {e}")


def check_and_enforce_quota():
    """
    Check quota limits and enforce rate limiting.
    Returns True if request can proceed, False if quota exceeded.
    """
    # Load state from file
    state = load_quota_state()
    daily_request_count = state["daily_request_count"]
    daily_reset_date = state["daily_reset_date"]
    request_timestamps = state["request_timestamps"]
    
    # Parse reset date if it exists
    if daily_reset_date:
        daily_reset_date = datetime.fromisoformat(daily_reset_date).date()
    
    current_time = datetime.now()
    
    # Reset daily counter if it's a new day
    if daily_reset_date is None or current_time.date() > daily_reset_date:
        daily_request_count = 0
        daily_reset_date = current_time.date()
        request_timestamps = []  # Clear old timestamps on new day
        print(f"📅 Daily quota reset for {daily_reset_date}")
    
    # Check daily limit
    if daily_request_count >= MAX_REQUESTS_PER_DAY:
        print(f"❌ Daily quota exceeded: {daily_request_count}/{MAX_REQUESTS_PER_DAY} requests used today")
        print(f"⏳ Quota resets at midnight. Current time: {current_time.strftime('%H:%M:%S')}")
        return False
    
    # Remove timestamps older than 1 minute
    one_minute_ago = current_time - timedelta(minutes=1)
    request_timestamps = [ts for ts in request_timestamps if ts > one_minute_ago]
    
    # Check RPM limit
    if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        oldest_request = min(request_timestamps)
        wait_time = 60 - (current_time - oldest_request).total_seconds()
        if wait_time > 0:
            print(f"⏸️  RPM limit reached ({len(request_timestamps)}/{MAX_REQUESTS_PER_MINUTE})")
            print(f"⏳ Waiting {wait_time:.1f} seconds before next request...")
            time.sleep(wait_time + 1)  # Add 1 second buffer
            # Refresh current time after sleep
            current_time = datetime.now()
    
    # Enforce minimum time between requests (15 seconds for 4 RPM)
    if request_timestamps:
        last_request = max(request_timestamps)
        elapsed = (current_time - last_request).total_seconds()
        if elapsed < SECONDS_BETWEEN_REQUESTS:
            wait_time = SECONDS_BETWEEN_REQUESTS - elapsed
            print(f"⏳ Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            current_time = datetime.now()
    
    # Record this request
    request_timestamps.append(current_time)
    daily_request_count += 1
    
    print(f"📊 Quota status: {daily_request_count}/{MAX_REQUESTS_PER_DAY} daily requests | "
          f"{len(request_timestamps)}/{MAX_REQUESTS_PER_MINUTE} requests in last minute")
    
    # Save updated state to file
    save_quota_state(daily_request_count, daily_reset_date, request_timestamps)
    
    return True


def parse_resume_with_ai(client: genai.Client, resume_text):
    """
    Send resume text to an AI model and get structured information back.
    
    Args:
        resume_text (str): The plain text extracted from the resume
        
    Returns:
        dict: Structured resume information or None if quota exceeded
    """
    print("Processing resume with AI model...")
    
    # Check quota before making request
    if not check_and_enforce_quota():
        print("❌ Cannot process resume: quota limit exceeded")
        return None
    
    prompt = f"""Extract and return the structured resume information from the text below. Only use what is explicitly stated in the text and do not infer or invent any details.
    Resume text:
    {resume_text}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Updated from gemini-2.0-flash
            contents=prompt, 
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=models.Resume,
            )
        )
        
        print("✅ Resume parsed successfully")
        return response.text
        
    except Exception as e:
        print(f"❌ Error during AI parsing: {e}")
        
        # Decrement counter since request failed
        state = load_quota_state()
        state["daily_request_count"] = max(0, state["daily_request_count"] - 1)
        
        # Convert date back if it exists
        reset_date = state["daily_reset_date"]
        if reset_date:
            reset_date = datetime.fromisoformat(reset_date).date()
        
        # Convert timestamps back
        timestamps = [datetime.fromisoformat(ts) for ts in state.get("request_timestamps", [])]
        
        save_quota_state(state["daily_request_count"], reset_date, timestamps)
        raise
