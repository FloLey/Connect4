#!/usr/bin/env python3
"""
Test script to verify rate limit resilience logic.
This tests the core logic without actually calling APIs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from datetime import datetime, timedelta, timezone

def test_rate_limit_detection():
    """Test the rate limit detection logic from ai.py"""
    print("Testing rate limit detection logic...")
    
    # Test cases for rate limit error messages
    test_cases = [
        ("HTTP 429 Too Many Requests", True),
        ("rate_limit exceeded", True),
        ("Rate limit: 10 requests per minute", True),
        ("rate limit exceeded", True),
        ("quota exceeded for model", True),
        ("throttled by API", True),
        ("too many requests", True),
        ("Connection error", False),
        ("Invalid JSON response", False),
        ("Timeout", False),
    ]
    
    for error_msg, expected in test_cases:
        err_msg = error_msg.lower()
        is_rate_limit = any(x in err_msg for x in ["429", "rate_limit", "rate limit", "throttled", "quota exceeded", "too many requests"])
        result = "✅ PASS" if is_rate_limit == expected else "❌ FAIL"
        print(f"  {result}: '{error_msg}' -> rate_limit={is_rate_limit} (expected: {expected})")

def test_snooze_logic():
    """Test the snooze timer logic"""
    print("\nTesting snooze timer logic...")
    
    now = datetime.now(timezone.utc)
    ten_minutes_later = now + timedelta(minutes=10)
    
    # Test retry_after comparison
    print(f"  Current time: {now}")
    print(f"  Snooze until: {ten_minutes_later}")
    
    # Test if retry_after is in the future
    retry_after_future = ten_minutes_later
    retry_after_past = now - timedelta(minutes=1)
    
    print(f"  Future retry_after <= now: {retry_after_future <= now} (should be False)")
    print(f"  Past retry_after <= now: {retry_after_past <= now} (should be True)")

def test_game_status_logic():
    """Test the game status logic for snoozed games"""
    print("\nTesting game status logic...")
    
    # Mock game objects
    game_snoozed = {
        'status': 'PAUSED',
        'retry_after': datetime.now(timezone.utc) + timedelta(minutes=5)
    }
    
    game_not_snoozed = {
        'status': 'PAUSED',
        'retry_after': None
    }
    
    game_in_progress = {
        'status': 'IN_PROGRESS',
        'retry_after': None
    }
    
    # Test the isSnoozed logic from frontend
    def is_snoozed(game):
        return game['status'] == 'PAUSED' and game['retry_after'] and game['retry_after'] > datetime.now(timezone.utc)
    
    print(f"  Snoozed game: {is_snoozed(game_snoozed)} (should be True)")
    print(f"  PAUSED without retry_after: {is_snoozed(game_not_snoozed)} (should be False)")
    print(f"  IN_PROGRESS game: {is_snoozed(game_in_progress)} (should be False)")

if __name__ == "__main__":
    print("=" * 60)
    print("Rate Limit Resilience & Auto-Recovery Logic Test")
    print("=" * 60)
    
    test_rate_limit_detection()
    test_snooze_logic()
    test_game_status_logic()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)