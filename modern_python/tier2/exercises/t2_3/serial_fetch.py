"""T2-3 练习 1 起始代码：串行抓取，先跑出基线耗时。

练习目标：改为 httpx.AsyncClient + gather 并发，对比三种方式的耗时。
URL 用 httpbin 的 delay 端点（每请求约 1s 服务端延迟）；
网络不可用时把 fetch 换成 `await asyncio.sleep(1)` 的 mock 协程。
"""
import time

import httpx

URLS = [f"https://httpbin.org/delay/1?i={i}" for i in range(10)]


def fetch_all_serial() -> list[int]:
    sizes = []
    with httpx.Client(timeout=30) as client:
        for url in URLS:
            resp = client.get(url)
            sizes.append(len(resp.content))
    return sizes


if __name__ == "__main__":
    start = time.perf_counter()
    sizes = fetch_all_serial()
    elapsed = time.perf_counter() - start
    print(f"串行 10 个请求: {elapsed:.2f}s, 响应字节数: {sizes}")
