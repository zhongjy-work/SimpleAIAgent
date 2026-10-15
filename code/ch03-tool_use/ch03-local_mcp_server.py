# -*- coding: utf-8 -*-
"""
本地 MCP 服务(通过 stdio 通信)

用途:提供两个"无副作用"的演示工具
  1. add_numbers : 计算两个整数的和
  2. query_order : 查询预先准备好的模拟订单(静态数据,不涉及任何真实账户)

注意:stdio 模式下,标准输出(stdout)被用来传输 MCP 协议消息,
所以本文件里绝对不能用 print() 输出日志,调试信息一律写到标准错误(stderr)。
"""

import re
import sys

# 兼容不同版本的 mcp 库:
#   mcp 2.x 中服务类叫 MCPServer(mcp.server.mcpserver)
#   mcp 1.x 中服务类叫 FastMCP(mcp.server.fastmcp)
# 业务规则错误统一使用 ToolError:它表示"可预期的失败",
# 错误消息会原样返回给调用方(再由应用交给模型解释)。
try:
    from mcp.server.mcpserver import MCPServer as ServerClass
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    from mcp.server.fastmcp import FastMCP as ServerClass
    from mcp.server.fastmcp.exceptions import ToolError

# 创建 MCP 服务对象,名字会在握手时告诉客户端
mcp = ServerClass("demo-server")

# 模拟订单数据(静态数据,仅用于演示)
ORDERS = {
    "ORD-1001": {"status": "已发货", "item": "机械键盘", "amount": 399},
    "ORD-1002": {"status": "待付款", "item": "显示器支架", "amount": 159},
    "ORD-1003": {"status": "已签收", "item": "USB 扩展坞", "amount": 249},
}

# 加法允许的数值范围(业务规则)
MAX_ABS_VALUE = 1000000

# 订单号格式:ORD- 加 4 位数字(业务规则)
ORDER_ID_PATTERN = re.compile(r"^ORD-[0-9]{4}$")


def log_to_stderr(text):
    """把日志写到标准错误,避免污染 stdio 协议通道。"""
    sys.stderr.write("[server] " + text + "\n")
    sys.stderr.flush()


@mcp.tool()
def add_numbers(a: int, b: int) -> str:
    """计算两个整数的和。

    参数:
        a: 第一个整数,绝对值不能超过 1000000
        b: 第二个整数,绝对值不能超过 1000000
    返回:
        形如 "a + b = 结果" 的文本
    """
    log_to_stderr("收到 add_numbers 调用: a=" + str(a) + ", b=" + str(b))

    # 业务规则检查:数值范围
    if abs(a) > MAX_ABS_VALUE or abs(b) > MAX_ABS_VALUE:
        raise ToolError("数值超出允许范围,绝对值不能超过 " + str(MAX_ABS_VALUE))

    total = a + b
    return str(a) + " + " + str(b) + " = " + str(total)


@mcp.tool()
def query_order(order_id: str) -> str:
    """根据订单号查询模拟订单的状态、商品和金额。

    参数:
        order_id: 订单号,格式为 ORD- 加 4 位数字,例如 ORD-1001
    返回:
        包含订单状态、商品名称和金额(元)的文本
    """
    log_to_stderr("收到 query_order 调用: order_id=" + str(order_id))

    # 业务规则检查 1:订单号格式
    if ORDER_ID_PATTERN.match(order_id) is None:
        raise ToolError("订单号格式不正确,应为 ORD- 加 4 位数字,例如 ORD-1001")

    # 业务规则检查 2:订单必须存在
    if order_id not in ORDERS:
        raise ToolError("订单不存在: " + order_id)

    order = ORDERS[order_id]
    return (
        "订单 " + order_id
        + ",状态:" + order["status"]
        + ",商品:" + order["item"]
        + ",金额:" + str(order["amount"]) + " 元"
    )


if __name__ == "__main__":
    # 以 stdio 方式运行:从标准输入读取请求,向标准输出写回响应
    mcp.run(transport="stdio")
