# -*- coding: utf-8 -*-
"""
HTTP 版 MCP 服务(streamable-http 传输)

作用:把 web_api_server.py 提供的 HTTP 接口包装成 MCP 工具。
      应用(LLM 客户端)只和本服务的 MCP 接口通信,不直接访问订单 API。

调用链:
  LLM 客户端 --MCP(HTTP)--> 本服务 --HTTP 请求--> 订单 Web API

提供的工具:
  list_orders  -> GET  /api/orders
  get_order    -> GET  /api/orders/<订单号>
  cancel_order -> POST /api/orders/<订单号>/cancel

环境变量:
  MCP_PORT  本 MCP 服务的监听端口,默认 8001(访问地址 http://127.0.0.1:8001/mcp)
  API_PORT  订单 Web API 的端口,默认 8000
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

# 兼容不同版本的 mcp 库:
#   mcp 2.x 中服务类叫 MCPServer(mcp.server.mcpserver)
#   mcp 1.x 中服务类叫 FastMCP(mcp.server.fastmcp)
# 业务失败统一使用 ToolError:它表示"可预期的失败",
# 错误消息会原样返回给调用方(再由应用交给模型解释)。
try:
    from mcp.server.mcpserver import MCPServer as ServerClass
    from mcp.server.mcpserver.exceptions import ToolError
    IS_MCP_V2 = True
except ImportError:
    from mcp.server.fastmcp import FastMCP as ServerClass
    from mcp.server.fastmcp.exceptions import ToolError
    IS_MCP_V2 = False

HOST = "127.0.0.1"
MCP_PORT = int(os.environ.get("MCP_PORT", "8001"))
API_PORT = os.environ.get("API_PORT", "8000")
API_BASE = "http://127.0.0.1:" + API_PORT

# 调用订单 API 的超时时间(秒)
API_TIMEOUT = 5

# 创建 MCP 服务对象,名字会在握手时告诉客户端
mcp = ServerClass("order-web-mcp")


def log_to_stderr(text):
    """把日志写到标准错误。"""
    sys.stderr.write("[mcp] " + text + "\n")
    sys.stderr.flush()


def call_api(method, path, query):
    """向订单 Web API 发送 HTTP 请求,返回解析后的 JSON 数据(字典)。

    参数:
        method: "GET" 或 "POST"
        path:   接口路径,例如 /api/orders
        query:  查询参数字典,没有就传空字典
    出错时统一抛出 ToolError,消息里写明失败原因,方便模型向用户解释:
        - API 返回 4xx/5xx:带上 HTTP 状态码和接口给出的错误说明
        - API 无法连接:说明订单服务不可用
    """
    url = API_BASE + path
    if len(query) > 0:
        url = url + "?" + urllib.parse.urlencode(query)
    log_to_stderr("调用订单 API: " + method + " " + url)

    data = None
    if method == "POST":
        data = b""
    request = urllib.request.Request(url, data=data, method=method)

    try:
        response = urllib.request.urlopen(request, timeout=API_TIMEOUT)
        body = response.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.HTTPError as error:
        # 接口返回了错误状态码,读取其中的错误说明
        message = "未知错误"
        try:
            error_body = json.loads(error.read().decode("utf-8"))
            message = error_body.get("error", message)
        except Exception:
            pass
        raise ToolError("订单服务返回错误(HTTP " + str(error.code) + "): " + message)
    except Exception as error:
        # 连接失败、超时等
        raise ToolError("无法连接订单服务: " + str(error))


def quote_order_id(order_id):
    """对订单号做 URL 编码后再拼进路径。
    这样即使参数里带有 / 或 .. 之类的字符,也不会改变请求的路径。"""
    return urllib.parse.quote(order_id, safe="")


@mcp.tool()
def list_orders(status: str = "") -> str:
    """查询订单列表,可以按订单状态过滤。

    参数:
        status: 可选。订单状态,只能是 待付款、已发货、已签收、已取消 之一;不传则返回全部订单
    返回:
        JSON 文本,包含订单数量 count 和订单数组 orders
    """
    query = {}
    if status != "":
        query["status"] = status
    result = call_api("GET", "/api/orders", query)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_order(order_id: str) -> str:
    """根据订单号查询单个订单的状态、商品和金额。

    参数:
        order_id: 订单号,格式为 ORD- 加 4 位数字,例如 ORD-1001
    返回:
        JSON 文本,包含订单号、状态、商品名称和金额(元)
    """
    result = call_api("GET", "/api/orders/" + quote_order_id(order_id), {})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def cancel_order(order_id: str) -> str:
    """取消一个订单。只有状态为"待付款"的订单可以取消。

    参数:
        order_id: 订单号,格式为 ORD- 加 4 位数字,例如 ORD-1002
    返回:
        JSON 文本,为取消后的订单信息(状态变为已取消)
    """
    result = call_api("POST", "/api/orders/" + quote_order_id(order_id) + "/cancel", {})
    return json.dumps(result, ensure_ascii=False)


def main():
    """以 streamable-http 方式启动 MCP 服务,一直运行直到进程被终止。"""
    log_to_stderr("MCP 服务启动: http://" + HOST + ":" + str(MCP_PORT) + "/mcp ,订单 API 地址: " + API_BASE)
    if IS_MCP_V2:
        # mcp 2.x:监听地址在 run 里传入
        mcp.run(transport="streamable-http", host=HOST, port=MCP_PORT)
    else:
        # mcp 1.x:监听地址写在 settings 里
        mcp.settings.host = HOST
        mcp.settings.port = MCP_PORT
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
