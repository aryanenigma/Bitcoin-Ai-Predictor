from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os

app = FastAPI()
app.mount("/analysis", StaticFiles(directory="analysis"), name="analysis")

@app.get("/stats")
def stats():
    if not os.path.exists("analysis/trades_log.csv"):
        return {"balance": 0, "winrate": 0}

    df = pd.read_csv("analysis/trades_log.csv")
    winrate = round((df[df.pnl_pct > 0].shape[0] / len(df)) * 100, 2)
    balance = round(10000 * (1 + df.pnl_pct/100).prod(), 2)

    return {
        "balance": balance,
        "winrate": winrate,
        "best": df.pnl_pct.max(),
        "top_trades": df.sort_values("pnl_pct", ascending=False).head(5).to_dict(orient="records")
    }
