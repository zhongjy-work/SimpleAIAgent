# -*- coding: utf-8 -*-
"""
简单的订单 Web API 服务(只使用 Python 标准库,无需额外安装依赖)

它代表"已经存在的业务系统",后面的 MCP 服务会把这里的 HTTP 接口包装成工具。
数据全部保存在内存中,服务重启后恢复为初始数据,不涉及任何真实账户和外部副作用。

提供的接口:
  GET  /api/health                健康检查
  GET  /api/orders                查询订单列表,可选参数 ?status=订单状态
  GET  /api/orders/<订单号>        查询单个订单
  POST /api/orders/<订单号>/cancel 取消订单(只有"待付款"的订单可以取消)

出错时统一返回 JSON:{"error": "错误说明"},并使用合适的 HTTP 状态码:
  400 请求参数不合法, 404 资源不存在, 409 与当前业务状态冲突

环境变量:
  API_PORT  监听端口,默认 8000
"""

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

HOST = "127.0.0.1"
PORT = int(os.environ.get("API_PORT", "8000"))

# 模拟订单数据(保存在内存中)
ORDERS = {
    "ORD-1001": {"order_id": "ORD-1001", "status": "已发货", "item": "机械键盘", "amount": 399},
    "ORD-1002": {"order_id": "ORD-1002", "status": "待付款", "item": "显示器支架", "amount": 159},
    "ORD-1003": {"order_id": "ORD-1003", "status": "已签收", "item": "USB 扩展坞", "amount": 249},
    "ORD-1004": {"order_id": "ORD-1004", "status": "待付款", "item": "无线鼠标", "amount": 89},
    "ORD-1005": {"order_id": "ORD-1005", "status": "已发货", "item": "降噪耳机", "amount": 799},
    "ORD-1006": {"order_id": "ORD-1006", "status": "已取消", "item": "手机支架", "amount": 39},
}

# 系统支持的订单状态,用于校验查询条件
VALID_STATUS = ["待付款", "已发货", "已签收", "已取消"]

# 订单号格式:ORD- 加 4 位数字
ORDER_ID_PATTERN = re.compile(r"^ORD-[0-9]{4}$")


def split_path(path):
    """把 URL 路径按 / 切分,并去掉空片段。
    例如 "/api/orders/ORD-1001" 变成 ["api", "orders", "ORD-1001"]。"""
    parts = []
    for piece in path.split("/"):
        if piece != "":
            parts.append(piece)
    return parts


class OrderApiHandler(BaseHTTPRequestHandler):
    """处理每一个 HTTP 请求。HTTPServer 一次只处理一个请求,所以修改内存数据不需要加锁。"""

    def log_message(self, format, *args):
        """重写日志方法:把访问日志写到标准错误,并加上 [api] 前缀。"""
        sys.stderr.write("[api] " + (format % args) + "\n")
        sys.stderr.flush()

    def send_json(self, status_code, data):
        """把 data 转成 JSON 并作为响应发送给调用方。"""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status_code, message):
        """发送统一格式的错误响应:{"error": 错误说明}。"""
        self.send_json(status_code, {"error": message})

    def do_GET(self):
        """处理所有 GET 请求,按路径分发到对应的处理函数。"""
        parsed = urlparse(self.path)
        parts = split_path(parsed.path)

        if parts == ["api", "health"]:
            self.send_json(200, {"status": "ok"})
        elif parts == ["api", "orders"]:
            self.handle_list_orders(parsed.query)
        elif len(parts) == 3 and parts[0] == "api" and parts[1] == "orders":
            self.handle_get_order(parts[2])
        else:
            self.send_error_json(404, "接口不存在: " + parsed.path)

    def do_POST(self):
        """处理所有 POST 请求,目前只有"取消订单"。"""
        parsed = urlparse(self.path)
        parts = split_path(parsed.path)

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "orders" and parts[3] == "cancel":
            self.handle_cancel_order(parts[2])
        else:
            self.send_error_json(404, "接口不存在: " + parsed.path)

    def handle_list_orders(self, query_text):
        """查询订单列表。可以通过 ?status=xxx 按状态过滤。"""
        query = parse_qs(query_text)
        status = ""
        if "status" in query:
            status = query["status"][0]

        # 传了 status 就必须是合法的状态值
        if status != "" and status not in VALID_STATUS:
            self.send_error_json(400, "不支持的订单状态: " + status + ",可选值: " + "、".join(VALID_STATUS))
            return

        result = []
        for order_id in ORDERS:
            order = ORDERS[order_id]
            if status == "" or order["status"] == status:
                result.append(order)
        self.send_json(200, {"count": len(result), "orders": result})

    def handle_get_order(self, order_id):
        """查询单个订单:先校验订单号格式,再检查订单是否存在。"""
        if ORDER_ID_PATTERN.match(order_id) is None:
            self.send_error_json(400, "订单号格式不正确,应为 ORD- 加 4 位数字,例如 ORD-1001")
            return
        if order_id not in ORDERS:
            self.send_error_json(404, "订单不存在: " + order_id)
            return
        self.send_json(200, ORDERS[order_id])

    def handle_cancel_order(self, order_id):
        """取消订单:只有"待付款"的订单可以取消,其它状态返回 409。"""
        if ORDER_ID_PATTERN.match(order_id) is None:
            self.send_error_json(400, "订单号格式不正确,应为 ORD- 加 4 位数字,例如 ORD-1001")
            return
        if order_id not in ORDERS:
            self.send_error_json(404, "订单不存在: " + order_id)
            return

        order = ORDERS[order_id]
        if order["status"] != "待付款":
            self.send_error_json(409, "只有待付款的订单可以取消,订单 " + order_id + " 当前状态为:" + order["status"])
            return

        order["status"] = "已取消"
        self.send_json(200, order)


def main():
    """启动 Web API 服务,一直运行直到进程被终止。"""
    server = HTTPServer((HOST, PORT), OrderApiHandler)
    sys.stderr.write("[api] 订单 Web API 已启动: http://" + HOST + ":" + str(PORT) + "\n")
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
