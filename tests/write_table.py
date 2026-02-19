import os
from src.config import Config
from src.utils.match_equity import MatchEquityTable

def main():
    equity_table = MatchEquityTable(Config.MATCH_TARGET)
    equity_path = os.path.join(Config.CHECKPOINT_DIR, 'match_equity.pt')
    equity_table.load(equity_path)
    equity_table.print_table()
    pass

if __name__=="__main__":
    main()