"""
Asyncio Event Loop 学习示例
展示如何获取、使用和管理事件循环
"""

import asyncio
import time


# 示例1: 基本的事件循环获取方式
def basic_event_loop_example():
    """演示获取事件循环的基本方法"""
    print("=== 基本事件循环示例 ===")
    
    # 获取当前事件循环（如果不存在会创建一个新的）
    try:
        loop = asyncio.get_event_loop()
        print(f"当前事件循环: {loop}")
    except RuntimeError as e:
        print(f"获取事件循环时出错: {e}")
    
    # 推荐的方式：获取事件循环或创建新的事件循环
    loop = asyncio.get_event_loop_policy().get_event_loop()
    print(f"通过策略获取的事件循环: {loop}")


# 示例2: 异步函数定义
async def simple_async_task(name, delay):
    """简单的异步任务"""
    print(f"任务 {name} 开始执行")
    await asyncio.sleep(delay)
    print(f"任务 {name} 执行完成，耗时 {delay} 秒")
    return f"任务 {name} 的结果"


# 示例3: 在事件循环中运行协程
async def run_coroutines_example():
    """演示在事件循环中运行协程"""
    print("\n=== 运行协程示例 ===")
    
    # 方法1: 使用 asyncio.run() (推荐用于程序主入口点)
    print("使用 asyncio.run() 运行单个协程:")
    result = await simple_async_task("A", 1)
    print(f"结果: {result}")
    
    # 方法2: 同时运行多个协程
    print("\n同时运行多个协程:")
    tasks = [
        simple_async_task("B1", 1),
        simple_async_task("B2", 2),
        simple_async_task("B3", 1.5)
    ]
    results = await asyncio.gather(*tasks)
    print(f"所有任务的结果: {results}")


# 示例4: 手动管理事件循环
def manual_event_loop_management():
    """演示手动管理事件循环"""
    print("\n=== 手动管理事件循环示例 ===")
    
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    print(f"创建了新的事件循环: {loop}")
    
    try:
        # 设置为当前线程的事件循环
        asyncio.set_event_loop(loop)
        print("已设置为当前线程的事件循环")
        
        # 在事件循环中运行协程
        result = loop.run_until_complete(simple_async_task("C", 1))
        print(f"手动运行结果: {result}")
        
        # 运行多个任务
        tasks = [
            simple_async_task("D1", 1),
            simple_async_task("D2", 1.5)
        ]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        print(f"手动运行多个任务的结果: {results}")
        
    finally:
        # 清理事件循环
        loop.close()
        print("事件循环已关闭")


# 示例5: 事件循环中的回调函数
async def callback_example():
    """演示事件循环中的回调"""
    print("\n=== 回调函数示例 ===")
    
    loop = asyncio.get_running_loop()
    
    def my_callback(arg1, arg2):
        print(f"回调函数被调用: {arg1}, {arg2}")
        return arg1 + arg2
    
    # 调度回调函数
    future = loop.run_in_executor(None, my_callback, "Hello", " World")
    result = await future
    print(f"回调函数返回结果: {result}")


# 示例6: 自定义事件循环策略
class CustomEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """自定义事件循环策略"""
    
    def new_event_loop(self):
        print("创建自定义事件循环")
        return super().new_event_loop()


def custom_policy_example():
    """演示自定义事件循环策略"""
    print("\n=== 自定义事件循环策略示例 ===")
    
    # 保存原始策略
    old_policy = asyncio.get_event_loop_policy()
    
    try:
        # 设置自定义策略
        asyncio.set_event_loop_policy(CustomEventLoopPolicy())
        print("设置了自定义事件循环策略")
        
        # 创建事件循环
        loop = asyncio.new_event_loop()
        print(f"使用自定义策略创建的事件循环: {loop}")
        
        # 运行简单任务
        result = loop.run_until_complete(simple_async_task("E", 1))
        print(f"自定义策略下运行结果: {result}")
        
        loop.close()
        
    finally:
        # 恢复原始策略
        asyncio.set_event_loop_policy(old_policy)
        print("恢复了原始事件循环策略")


# 示例7: 事件循环中的定时器
async def timer_example():
    """演示事件循环中的定时器功能"""
    print("\n=== 定时器示例 ===")
    
    loop = asyncio.get_running_loop()
    
    def timeout_callback():
        print("定时器回调被执行!")
    
    # 添加一个2秒后执行的定时器
    handle = loop.call_later(2, timeout_callback)
    print("已添加2秒后的定时器")
    
    # 取消定时器
    # handle.cancel()
    # print("定时器已被取消")
    
    # 等待定时器执行
    await asyncio.sleep(3)
    print("定时器示例结束")


# 主函数
async def main():
    """主函数，依次运行各个示例"""
    print("Asyncio Event Loop 学习示例开始\n")
    
    # 基本事件循环示例
    basic_event_loop_example()
    
    # 运行协程示例
    await run_coroutines_example()
    
    # 回调函数示例
    await callback_example()
    
    # 定时器示例
    await timer_example()
    
    print("\n=== 所有异步示例完成 ===")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
    
    # 手动管理事件循环示例（需要在asyncio.run之后单独运行）
    manual_event_loop_management()
    
    # 自定义策略示例
    custom_policy_example()
    
    print("\n所有示例运行完毕!")