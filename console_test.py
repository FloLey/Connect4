import sys
import os

# Ensure app module is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.app.engine.game import ConnectFour
from backend.app.engine.ai import ConnectFourAI

def main():
    print("=======================================")
    print("   CONNECT FOUR: Human vs GPT-4o")
    print("=======================================")
    
    game = ConnectFour()
    
    # Configure AI as Player 2
    ai_agent = ConnectFourAI(player_id=2)
    
    print(game.get_visual_board())

    while game.winner is None and not game.is_draw():
        
        # --- Human Turn (Player 1) ---
        if game.current_turn == 1:
            valid_moves = game.get_valid_moves()
            try:
                user_input = input(f"\nYour Move (Columns {valid_moves}): ")
                col = int(user_input)
                if col not in valid_moves:
                    print("Invalid column. Try again.")
                    continue
                
                game.drop_piece(col)
            except ValueError:
                print("Please enter a valid number.")
                continue

        # --- AI Turn (Player 2) ---
        else:
            print("\nAI is thinking...")
            try:
                # Call the LangChain Agent
                decision = ai_agent.get_move(game)
                
                print(f"AI Reasoning: {decision.reasoning}")
                print(f"AI plays Column: {decision.column}")
                
                game.drop_piece(decision.column)
            except Exception as e:
                print(f"AI Error: {e}")
                break

        # Show Board
        print("\n" + game.get_visual_board())

    # --- End Game ---
    if game.winner:
        winner_name = "Human" if game.winner == 1 else "AI"
        print(f"\nGame Over! Winner: {winner_name}")
    else:
        print("\nGame Over! It's a Draw.")

if __name__ == "__main__":
    main()