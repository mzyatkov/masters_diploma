# pip install requests beautifulsoup4
import re, time, random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}

def _extract_price(text: str) -> Optional[float]:
    """
    Получает '1,234.56' → 1234.56.  Вернёт None, если не найдено.
    """
    m = re.search(r"(\d[\d,]*)(?:\.(\d\d))?", text)
    if not m:
        return None
    whole = m.group(1).replace(",", "")
    frac  = m.group(2) or "00"
    return float(f"{whole}.{frac}")

# ------------------------------------------------------------------------
# 1. Amazon
# ------------------------------------------------------------------------
def _price_from_amazon(query: str) -> Optional[float]:
    url = f"https://www.amazon.com/s?k={requests.utils.quote(query)}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    # Берём первый блок цены целая+дробная
    price_whole = soup.select_one("span.a-price-whole")
    price_frac  = soup.select_one("span.a-price-fraction")
    if price_whole and price_frac:
        return _extract_price(price_whole.text + "." + price_frac.text)
    return None

# ------------------------------------------------------------------------
# 2. eBay
# ------------------------------------------------------------------------
def _price_from_ebay(query: str) -> Optional[float]:
    url = f"https://www.ebay.com/sch/i.html?_nkw={requests.utils.quote(query)}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    price_tag = soup.select_one("span.s-item__price")
    if price_tag:
        return _extract_price(price_tag.text)
    return None

# ------------------------------------------------------------------------
# 3. Public helper
# ------------------------------------------------------------------------
def find_disk_price(model: str, manufacturer: str) -> Optional[float]:
    """
    Returns minimum price in USD or None if nothing found.
    """
    query = f"{manufacturer} {model}"
    sources = {
        "amazon": _price_from_amazon,
        # "ebay":   _price_from_ebay,
    }

    prices = []
    for name, fn in sources.items():
        try:
            price = fn(query)
            if price:
                print(f"{name:<6} | {query:<30} → {price:>8.2f} USD")
                prices.append(price)
        except Exception as exc:
            print(f"{name:<6} | error: {exc}")

        # вежливая пауза 1-2 сек между площадками
        time.sleep(random.uniform(1.0, 2.0))

    return min(prices) if prices else None


# ---------------- Demo ----------------
if __name__ == "__main__":
    price = find_disk_price("ST12000NM001G", "Seagate")
    print("Lowest found:", price)

    df = pd.read_csv("output/all.csv")
    df["disk_price"] = df.apply(
        lambda row: find_disk_price(row["model"], row["type"] + " " + row["mfg"]), axis=1
    )
    print(df[["model", "manufacturer", "disk_price"]].head(10))
    df.to_csv("output/all_with_prices.csv")
