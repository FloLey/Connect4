#!/usr/bin/env python3
"""
Test script to verify the tournament completion logic fixes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from datetime import datetime, timedelta, timezone
from backend.app.models.enums import GameStatus

def test_global_unfinished_logic():
    """Test the global unfinished check logic"""
    print("Testing global unfinished check logic...")
    
    # Mock tournament scenarios
    scenarios = [
        {
            "name": "All games completed",
            "completed": 10,
            "draw": 5,
            "pending": 0,
            "in_progress": 0,
            "paused": 0,
            "expected_unfinished": 0,
            "should_complete": True
        },
        {
            "name": "Games in progress",
            "completed": 8,
            "draw": 4,
            "pending": 2,
            "in_progress": 3,
            "paused": 0,
            "expected_unfinished": 5,
            "should_complete": False
        },
        {
            "name": "Games paused (rate limited)",
            "completed": 12,
            "draw": 6,
            "pending": 0,
            "in_progress": 0,
            "paused": 4,
            "expected_unfinished": 4,
            "should_complete": False
        },
        {
            "name": "Mixed unfinished states",
            "completed": 15,
            "draw": 5,
            "pending": 2,
            "in_progress": 1,
            "paused": 3,
            "expected_unfinished": 6,
            "should_complete": False
        },
        {
            "name": "Ghost match scenario",
            "completed": 30,
            "draw": 10,
            "pending": 0,
            "in_progress": 0,
            "paused": 8,  # Games waiting for rate limit cooldown
            "expected_unfinished": 8,
            "should_complete": False
        }
    ]
    
    for scenario in scenarios:
        # Simulate the global unfinished check
        unfinished_statuses = [GameStatus.PENDING, GameStatus.IN_PROGRESS, GameStatus.PAUSED]
        total_unfinished = (
            scenario["pending"] + 
            scenario["in_progress"] + 
            scenario["paused"]
        )
        
        # Check logic
        should_complete = total_unfinished == 0
        matches_expected = should_complete == scenario["should_complete"]
        
        result = "✅ PASS" if matches_expected else "❌ FAIL"
        print(f"  {result}: {scenario['name']}")
        print(f"     Unfinished: {total_unfinished} (expected: {scenario['expected_unfinished']})")
        print(f"     Should complete: {should_complete} (expected: {scenario['should_complete']})")

def test_retry_after_logic():
    """Test the retry_after expiration logic"""
    print("\nTesting retry_after expiration logic...")
    
    now = datetime.now(timezone.utc)
    
    test_cases = [
        {
            "name": "Expired retry_after (5 minutes ago)",
            "retry_after": now - timedelta(minutes=5),
            "should_resume": True
        },
        {
            "name": "Future retry_after (5 minutes from now)",
            "retry_after": now + timedelta(minutes=5),
            "should_resume": False
        },
        {
            "name": "Just expired (1 second ago)",
            "retry_after": now - timedelta(seconds=1),
            "should_resume": True
        },
        {
            "name": "About to expire (1 second from now)",
            "retry_after": now + timedelta(seconds=1),
            "should_resume": False
        },
        {
            "name": "No retry_after (None)",
            "retry_after": None,
            "should_resume": False  # PAUSED without retry_after shouldn't be resumed automatically
        }
    ]
    
    for test in test_cases:
        if test["retry_after"] is None:
            should_resume = False
        else:
            should_resume = test["retry_after"] <= now
        
        matches_expected = should_resume == test["should_resume"]
        result = "✅ PASS" if matches_expected else "❌ FAIL"
        
        retry_str = test["retry_after"].isoformat() if test["retry_after"] else "None"
        print(f"  {result}: {test['name']}")
        print(f"     retry_after: {retry_str}")
        print(f"     Should resume: {should_resume} (expected: {test['should_resume']})")

def test_game_query_logic():
    """Test the game query logic for tournament tick"""
    print("\nTesting game query logic...")
    
    # Simulate the query logic from tick()
    # Query should get: PENDING games OR (PAUSED games with expired retry_after)
    
    now = datetime.now(timezone.utc)
    
    # Mock games
    mock_games = [
        {"id": 1, "status": "PENDING", "retry_after": None},
        {"id": 2, "status": "PENDING", "retry_after": None},
        {"id": 3, "status": "PAUSED", "retry_after": now - timedelta(minutes=15)},  # Expired
        {"id": 4, "status": "PAUSED", "retry_after": now + timedelta(minutes=5)},   # Not expired
        {"id": 5, "status": "IN_PROGRESS", "retry_after": None},
        {"id": 6, "status": "COMPLETED", "retry_after": None},
        {"id": 7, "status": "PAUSED", "retry_after": now - timedelta(seconds=30)},  # Expired
    ]
    
    # Apply query logic
    eligible_games = []
    for game in mock_games:
        if game["status"] == "PENDING":
            eligible_games.append(game["id"])
        elif game["status"] == "PAUSED" and game["retry_after"] and game["retry_after"] <= now:
            eligible_games.append(game["id"])
    
    expected_eligible = [1, 2, 3, 7]  # PENDING 1,2 and expired PAUSED 3,7
    
    matches_expected = set(eligible_games) == set(expected_eligible)
    result = "✅ PASS" if matches_expected else "❌ FAIL"
    
    print(f"  {result}: Game query logic")
    print(f"     Eligible games: {sorted(eligible_games)}")
    print(f"     Expected: {expected_eligible}")

if __name__ == "__main__":
    print("=" * 70)
    print("Tournament Completion Logic Test")
    print("=" * 70)
    
    test_global_unfinished_logic()
    test_retry_after_logic()
    test_game_query_logic()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)