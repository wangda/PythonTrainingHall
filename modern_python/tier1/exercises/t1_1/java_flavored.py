"""T1-1 练习 1 起始代码：补全类型标注直到 pyright strict 通过。

规则：只允许加标注和 import，禁止改运行逻辑。
"""


class ShoppingCartItem:
    def __init__(self, name: str, price: float, quantity: int):
        self.name: str = name
        self.price: float = price
        self.quantity: int = quantity


class ShoppingCart:
    def __init__(self):
        self.items : list[ShoppingCartItem] = []

    def add_item(self, item: ShoppingCartItem):
        self.items.append(item)

    def total(self) -> float:
        result: float = 0.0
        for item in self.items:
            result += item.price * item.quantity
        return result

    def find_by_name(self, name: str) -> ShoppingCartItem | None:
        for item in self.items:
            if item.name == name:
                return item
        return None

    def discounted_amounts(self, rate: float | int):
        if rate > 1 or rate < 0:
            raise ValueError("rate must be in [0, 1]")
        return [item.price * item.quantity * rate for item in self.items]


def summarize(cart: ShoppingCart, sep: str=", ") -> str:
    names: list[str] = []
    for item in cart.items:
        names.append(item.name)
    return sep.join(names)


def cheapest_item(cart: ShoppingCart) -> ShoppingCartItem | None:
    if not cart.items:
        return None
    best: ShoppingCartItem = cart.items[0]
    for item in cart.items:
        if item.price < best.price:
            best = item
    return best


if __name__ == "__main__":
    cart: ShoppingCart = ShoppingCart()
    cart.add_item(ShoppingCartItem("apple", 5.0, 3))
    cart.add_item(ShoppingCartItem("banana", 2.5, 6))
    print(summarize(cart))
    print(cart.total())
    print(cart.discounted_amounts(0.8))
    print(cheapest_item(cart))
    print(cart.find_by_name("apple") is not None)
    print(cart.find_by_name("cherry"))
