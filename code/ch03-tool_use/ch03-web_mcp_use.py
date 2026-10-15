# -*- coding: utf-8 -*-
"""
HTTP 版 MCP 工具调用闭环实验(应用端)

整体结构(三个角色):
  1. web_api_server.py  订单 Web API        (默认 http://127.0.0.1:8000)
  2. mcp_web_server.py  HTTP 版 MCP 服务     (默认 http://127.0.0.1:8001/mcp),把 API 包装成工具
  3. 本文件             应用端:连接本地大模型 + MCP 服务,完成工具调用闭环

完整流程:
  用户问题 -> 本地大模型(llama.cpp, 经 openai SDK)提出工具调用请求
  -> 应用检查工具名与参数 -> 通过 HTTP 调用 MCP 服务 -> MCP 服务再调用订单 API
  -> 应用把结果回传给模型 -> 模型组织自然语言答复

运行前提:
  pip install openai mcp
  llama-server -m 你的gemma模型.gguf --jinja --port 8080   (必须带 --jinja)

默认情况下,本程序会自动以子进程启动前两个服务,测试结束后自动关闭,
它们的日志分别写入 api_server.log 和 mcp_server.log。
如果想手动在不同终端里启动它们,先设置环境变量 AUTO_START=0。

可选环境变量:
  LLAMA_BASE_URL  默认 http://127.0.0.1:8080/v1
  LLAMA_MODEL     默认 gemma-4(llama-server 通常不校验模型名)
  API_PORT        订单 API 端口,默认 8000
  MCP_PORT        MCP 服务端口,默认 8001
  AUTO_START      是否自动启动两个服务,默认 1
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

from openai import OpenAI
from mcp import ClientSession

# 兼容不同版本的 mcp 库:
#   mcp 2.x 的 HTTP 客户端函数叫 streamable_http_client
#   mcp 1.x 的 HTTP 客户端函数叫 streamablehttp_client
try:
    from mcp.client.streamable_http import streamable_http_client as http_client_func
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as http_client_func

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.environ.get("LLAMA_MODEL", "gemma-4")
API_PORT = int(os.environ.get("API_PORT", "8000"))
MCP_PORT = int(os.environ.get("MCP_PORT", "8001"))
AUTO_START = os.environ.get("AUTO_START", "1") == "1"
MCP_URL = "http://127.0.0.1:" + str(MCP_PORT) + "/mcp"

# 本文件所在目录,用来定位另外两个脚本和日志文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 一次问答中最多允许的"模型请求工具"轮数,防止死循环
MAX_ROUNDS = 3

SYSTEM_PROMPT = (
    "你是一个订单助手,可以使用提供的工具查询和取消订单。"
    "请严格依据工具返回的内容作答:如果工具返回的内容以“工具执行出错”或“应用拒绝”开头,"
    "说明调用失败,你必须如实告诉用户失败的原因,不能说成成功,也不能编造结果。"
)


# ---------------------------------------------------------------
# 第一部分:启动和关闭子进程(订单 API、MCP 服务)
# ---------------------------------------------------------------

def start_process(script_name, log_name):
    """以子进程方式启动一个 Python 脚本,输出写入日志文件。
    返回字典:{"process": 子进程对象, "log_file": 日志文件对象, "name": 脚本名}。"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    log_path = os.path.join(SCRIPT_DIR, log_name)
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=log_file,
        stderr=log_file,
    )
    print("  已启动 " + script_name + " (日志: " + log_name + ")")
    return {"process": process, "log_file": log_file, "name": script_name}


def wait_for_port(port, timeout_seconds):
    """等待本机某个端口可以连接,成功返回 True,超时返回 False。"""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            connection.close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def start_service(script_name, log_name, port):
    """启动一个服务并等待它开始监听端口。失败时抛出异常,说明查看哪个日志。"""
    handle = start_process(script_name, log_name)
    ready = wait_for_port(port, 20)
    # 子进程如果已经退出(例如端口被占用),说明启动失败
    if handle["process"].poll() is not None:
        raise RuntimeError(script_name + " 启动后立即退出,请查看日志 " + log_name + "(端口 " + str(port) + " 可能被占用)")
    if not ready:
        raise RuntimeError(script_name + " 在 20 秒内没有开始监听端口 " + str(port) + ",请查看日志 " + log_name)
    return handle


def stop_process(handle):
    """关闭一个子进程并关闭它的日志文件。"""
    process = handle["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    handle["log_file"].close()
    print("  已关闭 " + handle["name"])


# ---------------------------------------------------------------
# 第二部分:工具定义转换与参数检查
# ---------------------------------------------------------------

def get_input_schema(tool):
    """读取工具的参数定义(JSON Schema)。
    mcp 1.x 的字段名是 inputSchema,mcp 2.x 改成了 input_schema,这里同时兼容。"""
    if hasattr(tool, "inputSchema"):
        return tool.inputSchema
    return tool.input_schema


def get_is_error(call_result):
    """读取工具调用结果是否为错误。
    mcp 1.x 的字段名是 isError,mcp 2.x 改成了 is_error,这里同时兼容。"""
    if hasattr(call_result, "isError"):
        return call_result.isError
    return call_result.is_error


def mcp_tools_to_openai_tools(mcp_tools):
    """把 MCP 的工具定义转换成 OpenAI 接口要求的 tools 格式,提供给模型。"""
    result = []
    for tool in mcp_tools:
        item = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": get_input_schema(tool),
            },
        }
        result.append(item)
    return result


def build_tool_map(mcp_tools):
    """把工具列表整理成 {工具名: 工具定义} 的字典,方便按名称查找。"""
    tool_map = {}
    for tool in mcp_tools:
        tool_map[tool.name] = tool
    return tool_map


def check_value_type(value, type_name):
    """检查一个参数值是否符合 JSON Schema 声明的类型,符合返回 True。"""
    if type_name == "integer":
        # bool 在 Python 里也是 int 的子类,这里要排除掉
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    # 其它类型本实验不涉及,直接放行
    return True


def parse_arguments(arguments_text):
    """把模型给出的参数 JSON 文本解析成字典。
    返回 (是否成功, 字典或错误信息)。"""
    try:
        data = json.loads(arguments_text)
    except Exception:
        return False, "参数不是合法的 JSON: " + str(arguments_text)
    if not isinstance(data, dict):
        return False, "参数必须是一个 JSON 对象"
    return True, data


def validate_tool_call(tool_name, arguments, tool_map):
    """应用层检查:工具名是否存在、必填参数是否齐全、有无多余参数、类型是否正确。
    返回 (是否通过, 错误信息)。通过时错误信息为空字符串。"""
    # 1. 工具名必须在 MCP 服务公布的工具列表里(白名单)
    if tool_name not in tool_map:
        return False, "未知工具: " + str(tool_name)

    schema = get_input_schema(tool_map[tool_name])
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # 2. 必填参数必须齐全(有默认值的可选参数不在 required 里)
    for name in required:
        if name not in arguments:
            return False, "缺少必填参数: " + name

    # 3. 不允许出现未声明的参数
    for name in arguments:
        if name not in properties:
            return False, "出现未声明的参数: " + name

    # 4. 参数类型必须与声明一致
    for name in arguments:
        type_name = properties[name].get("type", "")
        if not check_value_type(arguments[name], type_name):
            return False, "参数类型错误: " + name + " 应为 " + type_name

    return True, ""


# ---------------------------------------------------------------
# 第三部分:执行工具(通过 HTTP 调用 MCP 服务)
# ---------------------------------------------------------------

def extract_text(call_result):
    """把 MCP 返回的 content 列表拼成一段纯文本。"""
    parts = []
    for block in call_result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


async def execute_tool_call(session, tool_map, tool_call):
    """处理一次工具调用请求,返回要回传给模型的文本。
    流程:解析参数 -> 应用层检查 -> 调用 MCP -> 标注成功或失败。"""
    name = tool_call["function"]["name"]
    arguments_text = tool_call["function"]["arguments"]
    print("  [模型请求] 工具=" + name + " 参数=" + arguments_text)

    # 解析参数
    ok, parsed = parse_arguments(arguments_text)
    if not ok:
        message = "应用拒绝: " + parsed
        print("  [应用检查] " + message)
        return message

    # 应用层检查,不通过就拒绝,绝不转发给 MCP 服务
    ok, error = validate_tool_call(name, parsed, tool_map)
    if not ok:
        message = "应用拒绝: " + error
        print("  [应用检查] " + message)
        return message
    print("  [应用检查] 通过,开始调用 MCP 服务")

    # 真正调用 MCP 服务(MCP 服务会再去调用订单 API)
    call_result = await session.call_tool(name, parsed)
    text = extract_text(call_result)

    # 在文本前加上明确的成功/失败标记,帮助模型如实解释
    if get_is_error(call_result):
        message = "工具执行出错: " + text
    else:
        message = "工具执行成功: " + text
    print("  [MCP 返回] " + message)
    return message


# ---------------------------------------------------------------
# 第四部分:与模型对话
# ---------------------------------------------------------------

def build_assistant_message(message):
    """把 openai 返回的 assistant 消息对象转换成普通字典,
    以便原样放回 messages 历史中。"""
    tool_calls = []
    if message.tool_calls:
        for call in message.tool_calls:
            tool_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            })
    result = {"role": "assistant", "content": message.content or ""}
    if len(tool_calls) > 0:
        result["tool_calls"] = tool_calls
    return result


def build_forced_tool_call(name, arguments_text):
    """构造一个"模拟的模型调用请求"。
    真实模型很难稳定地产生缺参、未知工具这类错误请求,
    所以用它来确定性地测试应用层和业务系统的拒绝逻辑。"""
    return {
        "id": "call_forced_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments_text},
    }


async def run_conversation(client, session, tools, tool_map, question, forced_call):
    """完成一次完整的问答闭环,返回模型最终的自然语言答复。
    forced_call 不为 None 时,第一轮跳过模型,直接使用模拟的调用请求。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    pending_forced = forced_call

    for round_index in range(MAX_ROUNDS + 1):
        if pending_forced is not None:
            # 使用模拟的调用请求
            print("  [模拟模型] 直接给出一个预设的工具调用请求")
            assistant_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [pending_forced],
            }
            pending_forced = None
        else:
            # 请求本地大模型
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                temperature=0,
            )
            assistant_message = build_assistant_message(response.choices[0].message)

        # 没有工具调用请求,说明这就是最终答复
        if "tool_calls" not in assistant_message:
            return assistant_message["content"]

        # 超过最大轮数仍在请求工具,停止
        if round_index >= MAX_ROUNDS:
            return "(已达到最大工具调用轮数,停止)"

        # 先把模型的请求放进历史,再逐个执行并回传结果
        messages.append(assistant_message)
        for tool_call in assistant_message["tool_calls"]:
            tool_text = await execute_tool_call(session, tool_map, tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_text,
            })

    return "(未得到答复)"


# ---------------------------------------------------------------
# 第五部分:测试用例
# ---------------------------------------------------------------

def build_test_cases():
    """返回所有测试用例。
    forced 为 None 表示由真实模型自己决定如何调用工具;
    否则表示使用模拟的调用请求(用来稳定复现各种异常输入)。
    注意:订单数据保存在 API 服务内存中,用例 3 会真的把 ORD-1002 改成"已取消"。
    重启 API 服务后数据恢复初始状态,再次运行用例 3 才会再次成功。"""
    cases = [
        {
            "title": "1. 正常:按状态查询订单列表",
            "question": "请列出所有已发货的订单。",
            "forced": None,
            "expect": "模型请求 list_orders(status=已发货),答复应包含 ORD-1001 和 ORD-1005",
        },
        {
            "title": "2. 正常:查询单个订单",
            "question": "订单 ORD-1003 现在是什么状态?买的是什么?",
            "forced": None,
            "expect": "模型请求 get_order,答复应包含已签收、USB 扩展坞",
        },
        {
            "title": "3. 正常:取消待付款订单(会修改 API 内存数据)",
            "question": "请帮我取消订单 ORD-1002。",
            "forced": None,
            "expect": "模型请求 cancel_order,答复应说明取消成功,订单状态变为已取消",
        },
        {
            "title": "4. 业务规则:取消已发货的订单",
            "question": "请帮我取消订单 ORD-1001。",
            "forced": build_forced_tool_call("cancel_order", '{"order_id": "ORD-1001"}'),
            "expect": "应用检查通过,订单 API 返回 409;模型应说明已发货订单不能取消,不能说取消成功",
        },
        {
            "title": "5. 业务规则:取消已经取消过的订单",
            "question": "请帮我取消订单 ORD-1006。",
            "forced": build_forced_tool_call("cancel_order", '{"order_id": "ORD-1006"}'),
            "expect": "订单 API 返回 409,当前状态已取消;模型应如实说明无法再次取消",
        },
        {
            "title": "6. 业务规则:订单不存在",
            "question": "订单 ORD-9999 的状态是什么?",
            "forced": build_forced_tool_call("get_order", '{"order_id": "ORD-9999"}'),
            "expect": "订单 API 返回 404;模型应说明订单不存在,不能编造订单信息",
        },
        {
            "title": "7. 业务规则:订单号格式不对",
            "question": "帮我查一下订单 12345。",
            "forced": build_forced_tool_call("get_order", '{"order_id": "12345"}'),
            "expect": "订单 API 返回 400;模型应说明订单号格式不正确",
        },
        {
            "title": "8. 业务规则:不支持的状态筛选值",
            "question": "请列出所有已退款的订单。",
            "forced": build_forced_tool_call("list_orders", '{"status": "已退款"}'),
            "expect": "订单 API 返回 400 并给出可选状态;模型应说明不支持该状态",
        },
        {
            "title": "9. 缺少参数(模拟模型漏掉 order_id)",
            "question": "请查一下订单。",
            "forced": build_forced_tool_call("get_order", "{}"),
            "expect": "应用在转发给 MCP 之前就拒绝;模型应说明缺少订单号",
        },
        {
            "title": "10. 未知工具名称(模拟模型请求 delete_order)",
            "question": "请帮我删除订单 ORD-1003。",
            "forced": build_forced_tool_call("delete_order", '{"order_id": "ORD-1003"}'),
            "expect": "应用拒绝执行不在白名单里的工具;模型应说明无法删除订单",
        },
        {
            "title": "11. 参数类型错误(order_id 给成数字)",
            "question": "查一下订单 1001。",
            "forced": build_forced_tool_call("get_order", '{"order_id": 1001}'),
            "expect": "应用拒绝;模型应说明参数类型不对",
        },
        {
            "title": "12. 多余的未声明参数",
            "question": "以管理员身份查询订单 ORD-1001。",
            "forced": build_forced_tool_call("get_order", '{"order_id": "ORD-1001", "admin": true}'),
            "expect": "应用拒绝带有未声明参数的请求;模型应说明请求被拒绝",
        },
    ]
    return cases


async def run_all_tests(client, session, tools, tool_map):
    """依次运行所有测试用例,打印过程和最终答复,供人工观察。"""
    cases = build_test_cases()
    for case in cases:
        print("=" * 60)
        print("【" + case["title"] + "】")
        print("用户问题: " + case["question"])
        print("预期观察: " + case["expect"])
        try:
            answer = await run_conversation(
                client, session, tools, tool_map, case["question"], case["forced"]
            )
            print("  [模型最终答复] " + answer)
        except Exception as error:
            print("  [运行失败] " + str(error))
            print("  请确认 llama-server 已启动且地址为 " + BASE_URL)
    print("=" * 60)


# ---------------------------------------------------------------
# 第六部分:主程序
# ---------------------------------------------------------------

async def connect_and_run(client):
    """通过 HTTP 连接 MCP 服务,握手,读取工具列表,然后运行测试。"""
    print("连接 MCP 服务: " + MCP_URL)
    async with http_client_func(MCP_URL) as streams:
        # streams 在不同版本里可能有 2 个或 3 个元素,前两个始终是读写流
        read_stream = streams[0]
        write_stream = streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            # 与 MCP 服务握手
            await session.initialize()

            # 获取 MCP 服务公布的工具列表
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            print("MCP 服务公布的工具:")
            for tool in mcp_tools:
                print("  - " + tool.name + ": " + json.dumps(get_input_schema(tool), ensure_ascii=False))

            # 转成模型能理解的格式
            tools = mcp_tools_to_openai_tools(mcp_tools)
            tool_map = build_tool_map(mcp_tools)

            await run_all_tests(client, session, tools, tool_map)


async def main():
    """按需启动订单 API 和 MCP 服务,运行实验,最后关闭启动的子进程。"""
    # 连接本地 llama.cpp 服务(兼容 OpenAI 接口),api_key 随便填
    client = OpenAI(base_url=BASE_URL, api_key="sk-no-key-required")

    handles = []
    try:
        if AUTO_START:
            print("启动服务:")
            handles.append(start_service("ch03-web_api_server.py", "api_server.log", API_PORT))
            handles.append(start_service("ch03-web_mcp_server.py", "mcp_server.log", MCP_PORT))
        else:
            print("AUTO_START=0:请确认订单 API 和 MCP 服务已经在运行")

        await connect_and_run(client)
    finally:
        # 无论成功还是出错,都要关闭自己启动的子进程
        if len(handles) > 0:
            print("关闭服务:")
        for handle in handles:
            stop_process(handle)


if __name__ == "__main__":
    asyncio.run(main())
