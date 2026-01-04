/**
 * Token Cleanup Utility
 * 
 * Cleans up old game tokens from localStorage to prevent accumulation.
 * Tokens older than 24 hours are removed.
 */

export const cleanupOldGameTokens = () => {
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000; // 24 hours in milliseconds
  let cleanedCount = 0;
  
  // Iterate through all localStorage keys
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    
    // Check if this is a game token key (pattern: game_{id}_token)
    if (key && key.startsWith('game_') && key.endsWith('_token')) {
      try {
        // Extract game ID from key
        const gameId = key.replace('game_', '').replace('_token', '');
        
        // Try to get the game creation time from another key
        // We'll store creation time when we save the token
        const creationKey = `game_${gameId}_created`;
        const creationTime = localStorage.getItem(creationKey);
        
        if (creationTime) {
          const age = now - parseInt(creationTime, 10);
          if (age > oneDayMs) {
            // Token is older than 24 hours, remove it
            localStorage.removeItem(key);
            localStorage.removeItem(creationKey);
            cleanedCount++;
          }
        } else {
          // No creation time stored, remove the token anyway
          localStorage.removeItem(key);
          cleanedCount++;
        }
      } catch (error) {
        console.warn(`Error cleaning up token ${key}:`, error);
      }
    }
  }
  
  if (cleanedCount > 0) {
    console.log(`Cleaned up ${cleanedCount} old game tokens`);
  }
  
  return cleanedCount;
};

/**
 * Enhanced token saving that also stores creation time
 */
export const saveGameTokenWithTimestamp = (gameId, token) => {
  localStorage.setItem(`game_${gameId}_token`, token);
  localStorage.setItem(`game_${gameId}_created`, Date.now().toString());
};

/**
 * Run cleanup on app startup (optional)
 */
export const initTokenCleanup = () => {
  // Run cleanup once per day
  const lastCleanup = localStorage.getItem('last_token_cleanup');
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;
  
  if (!lastCleanup || (now - parseInt(lastCleanup, 10)) > oneDayMs) {
    cleanupOldGameTokens();
    localStorage.setItem('last_token_cleanup', now.toString());
  }
};