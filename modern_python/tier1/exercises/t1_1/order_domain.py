from dataclasses import dataclass, field, replace
from enum import Enum, auto


class OrderStatus(Enum):
    PENDING = auto()
    PAID = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class OrderItem:
    sku: str
    unit_price: float
    quantity: int = 1


@dataclass()
class Order:
    order_id: str
    items: list[OrderItem] = field(default_factory=list[OrderItem])
    status: OrderStatus = OrderStatus.PENDING

    def total(self) -> float:
        return sum([item.unit_price * item.quantity for item in self.items])


def mark_paid(order: Order) -> Order:
    return replace(order, status=OrderStatus.PAID)


if __name__ == "__main__":
    order = Order("123", [OrderItem("a", 1.0, 2), OrderItem("b", 2.0, 3)])
    print(order.total())
    print(mark_paid(order).status)
