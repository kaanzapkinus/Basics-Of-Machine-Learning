import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://minecraft.fandom.com/wiki/Trading"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, "html.parser")

records = []
for tbl in soup.select("table.wikitable"):
    # En yakın önceki <h3> içinden meslek adını al
    prof = None
    for prev in tbl.find_all_previous():
        if prev.name == "h3":
            hl = prev.find("span", class_="mw-headline")
            prof = hl.text if hl else prev.get_text(strip=True)
            break
    # Tablo başlığı level (tier) bilgisi
    cap = tbl.find("caption")
    tier = cap.get_text(strip=True) if cap else ""
    # HTML tablosunu DataFrame’e çevir
    df = pd.read_html(str(tbl))[0]
    df["Profession"] = prof
    df["Tier"] = tier
    records.append(df)

full_df = pd.concat(records, ignore_index=True)
full_df.to_csv("villager_trades.csv", index=False)
print(full_df.head())