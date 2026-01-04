"""
Tournament Bus - Signal-based tournament tick system

Replaces the 2-second polling with an asyncio.Event-based system
to trigger tournament watcher immediately when a game slot becomes available.
"""

import asyncio


class TournamentBus:
    """Simple internal signal bus for tournament coordination"""
    def __init__(self):
        self.signal = asyncio.Event()

    def trigger(self):
        """Trigger the tournament watcher to run immediately"""
        self.signal.set()

    async def wait_for_signal(self, timeout: float = 30.0):
        """
        Wait for a signal or timeout
        
        Args:
            timeout: Maximum time to wait before returning (safety fallback)
        
        Returns:
            bool: True if signal was received, False if timed out
        """
        try:
            await asyncio.wait_for(self.signal.wait(), timeout=timeout)
            self.signal.clear()  # Reset after receiving signal
            return True
        except asyncio.TimeoutError:
            return False


# Global tournament bus instance
tournament_bus = TournamentBus()