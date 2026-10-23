"""T1-1 练习 1 起始代码：补全类型标注直到 pyright strict 通过。

规则：只允许加标注和 import，禁止改运行逻辑。
"""


class ShoppingCartItem:
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def total(self):
        result = 0.0
        for item in self.items:
            result += item.price * item.quantity
        return result

    def find_by_name(self, name):
        for item in self.items:
            if item.name == name:
                return item
        return None

    def discounted_amounts(self, rate):
        if rate > 1 or rate < 0:
            raise ValueError("rate must be in [0, 1]")
        return [item.price * item.quantity * rate for item in self.items]


def summarize(cart, sep=", "):
    names = []
    for item in cart.items:
        names.append(item.name)
    return sep.join(names)


def cheapest_item(cart):
    if not cart.items:
        return None
    best = cart.items[0]
    for item in cart.items:
        if item.price < best.price:
            best = item
    return best


if __name__ == "__main__":
    cart = ShoppingCart()
    cart.add_item(ShoppingCartItem("apple", 5.0, 3))
    cart.add_item(ShoppingCartItem("banana", 2.5, 6))
    print(summarize(cart))
    print(cart.total())
    print(cart.discounted_amounts(0.8))
    print(cheapest_item(cart))
    print(cart.find_by_name("apple") is not None)
    print(cart.find_by_name("cherry"))
