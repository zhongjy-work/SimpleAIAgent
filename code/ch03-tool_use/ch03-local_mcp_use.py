# -*- coding: utf-8 -*-
"""
MCP 工具调用闭环实验(应用端)

完整流程:
  用户问题 -> 本地大模型(llama.cpp, 经 openai SDK)提出工具调用请求
  -> 应用检查工具名与参数 -> 通过 stdio 调用本地 MCP 子进程
  -> 应用把结果回传给模型 -> 模型组织自然语言答复

说明:MCP 官方 Python SDK 是异步接口,所以本文件使用了 async / await,
      除此之外尽量只使用基础语法(函数、if、for、字典、列表)。

运行前提:
  pip install openai mcp
  llama-server -m 你的gemma模型.gguf --jinja --port 8080
  (必须带 --jinja,llama.cpp 才会启用模型的工具调用模板)

可选环境变量:
  LLAMA_BASE_URL  默认 http://127.0.0.1:8080/v1
  LLAMA_MODEL     默认 gemma-4(llama-server 通常不校验模型名)
"""

import asyncio
import json
import os
import sys

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.environ.get("LLAMA_MODEL", "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")

# 一次问答中最多允许的"模型请求工具"轮数,防止死循环
MAX_ROUNDS = 3

SYSTEM_PROMPT = (
    "你是一个助手,可以使用提供的工具回答问题。"
    "请严格依据工具返回的内容作答:如果工具返回的内容以“工具执行出错”或“应用拒绝”开头,"
    "说明调用失败,你必须如实告诉用户失败的原因,不能说成成功,也不能编造结果。"
)


# ---------------------------------------------------------------
# 第一部分:工具定义转换与参数检查
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

    # 2. 必填参数必须齐全
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
# 第二部分:执行工具(通过 stdio 调用 MCP 子进程)
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

    # 应用层检查,不通过就拒绝,绝不执行
    ok, error = validate_tool_call(name, parsed, tool_map)
    if not ok:
        message = "应用拒绝: " + error
        print("  [应用检查] " + message)
        return message
    print("  [应用检查] 通过,开始调用 MCP 服务")

    # 真正调用 MCP 子进程
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
# 第三部分:与模型对话
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
    所以用它来确定性地测试应用层的拦截逻辑。"""
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
# 第四部分:测试用例
# ---------------------------------------------------------------

def build_test_cases():
    """返回所有测试用例。
    forced 为 None 表示由真实模型自己决定如何调用工具;
    否则表示使用模拟的调用请求(用来稳定复现各种异常输入)。"""
    cases = [
        {
            "title": "1. 正常参数:加法",
            "question": "请帮我算一下 123 加 456 等于多少?",
            "forced": None,
            "expect": "模型请求 add_numbers,应用放行,答复应包含 579",
        },
        {
            "title": "2. 正常参数:查询订单",
            "question": "帮我查一下订单 ORD-1001 现在是什么状态?",
            "forced": None,
            "expect": "模型请求 query_order,答复应包含已发货、机械键盘、399",
        },
        {
            "title": "3. 缺少参数(模拟模型只给了 a)",
            "question": "请计算 5 加 7。",
            "forced": build_forced_tool_call("add_numbers", '{"a": 5}'),
            "expect": "应用在调用 MCP 之前就拒绝;模型应说明缺少参数,不能说算出了结果",
        },
        {
            "title": "4. 未知工具名称(模拟模型请求 delete_order)",
            "question": "请帮我删除订单 ORD-1001。",
            "forced": build_forced_tool_call("delete_order", '{"order_id": "ORD-1001"}'),
            "expect": "应用拒绝执行不在白名单里的工具;模型应说明无法删除",
        },
        {
            "title": "5. 参数类型错误(模拟模型把数字写成字符串)",
            "question": "请计算 abc 加 1。",
            "forced": build_forced_tool_call("add_numbers", '{"a": "abc", "b": 1}'),
            "expect": "应用拒绝;模型应说明参数类型不对",
        },
        {
            "title": "6. 业务规则:订单号格式不对",
            "question": "帮我查一下订单 12345。",
            "forced": build_forced_tool_call("query_order", '{"order_id": "12345"}'),
            "expect": "应用检查通过(类型正确),但 MCP 服务拒绝;模型应说明订单号格式不正确",
        },
        {
            "title": "7. 业务规则:订单不存在",
            "question": "帮我查一下订单 ORD-9999。",
            "forced": build_forced_tool_call("query_order", '{"order_id": "ORD-9999"}'),
            "expect": "MCP 服务返回错误;模型应说明订单不存在,不能编造订单信息",
        },
        {
            "title": "8. 业务规则:数值超出范围",
            "question": "请计算 5000000000 加 1。",
            "forced": build_forced_tool_call("add_numbers", '{"a": 5000000000, "b": 1}'),
            "expect": "MCP 服务返回超出范围的错误;模型应如实转述,不能给出求和结果",
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
# 第五部分:主程序
# ---------------------------------------------------------------

async def main():
    """启动 MCP 子进程,握手,读取工具列表,然后运行测试。"""
    # 用当前 Python 解释器以子进程方式启动 MCP 服务,通过 stdio 通信
    server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch03-local_mcp_server.py")
    server_params = StdioServerParameters(command=sys.executable, args=[server_path])

    # 连接本地 llama.cpp 服务(兼容 OpenAI 接口),api_key 随便填
    client = OpenAI(base_url=BASE_URL, api_key="sk-no-key-required")

    async with stdio_client(server_params) as streams:
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


if __name__ == "__main__":
    asyncio.run(main())
