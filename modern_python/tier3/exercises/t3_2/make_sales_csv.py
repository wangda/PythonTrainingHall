"""T3-2 练习 1 起始代码：生成带脏数据的销售 CSV。

列：order_id, region, category, price, qty, amount
脏数据：amount 为空串（null）、负数金额、category 乱码。
"""
import random
from pathlib import Path

REGIONS = ["north", "south", "east", "west"]
CATEGORIES = ["book", "toy", "food"]
BAD_CATEGORIES = ["", "BOOK", "toy "]  # 大小写/空格不一致，考察清洗策略

def main() -> None:
    out = Path(__file__).parent / "sales.csv"
    rng = random.Random(7)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("order_id,region,category,price,qty,amount\n")
        for i in range(100_000):
            region = rng.choice(REGIONS)
            category = rng.choice(CATEGORIES + BAD_CATEGORIES)
            price = round(rng.uniform(1, 200), 2)
            qty = rng.randint(1, 5)
            roll = rng.random()
            if roll < 0.03:
                amount = ""                      # 空值
            elif roll < 0.06:
                price = -price                   # 负数金额
                amount = f"{round(price * qty, 2)}"
            else:
                amount = f"{round(price * qty, 2)}"
            f.write(f"{i},{region},{category},{price},{qty},{amount}\n")
    print(f"written: {out} ({out.stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":
    main()
