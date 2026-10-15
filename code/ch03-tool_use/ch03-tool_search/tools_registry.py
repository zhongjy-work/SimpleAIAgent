"""模拟的工具注册表。

真实系统中，这些定义通常来自多个 MCP Server 的 tools/list 响应；
这里用代码生成约 40 个工具，覆盖不同领域，模拟"工具很多、单次任务只用到其中一两个"的场景。

每个工具包含：
  name / description / input_schema —— 会发送给模型的标准定义
  hint —— 仅用于检索索引（同义词、中文关键词），不会发给模型
"""

from __future__ import annotations


def _t(name, description, props, required, hint=""):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                k: {"type": v[0], "description": v[1]} for k, v in props.items()
            },
            "required": required,
        },
        "hint": hint,
    }


S = "string"
N = "number"

TOOLS: list[dict] = [
    # ---- 天气 ----
    _t("get_weather_forecast",
       "Get the weather forecast for a city for a given date. Returns temperature range, "
       "precipitation probability and a short text summary. Supports up to 7 days ahead.",
       {"city": (S, "City name, e.g. 'Tokyo'"),
        "date": (S, "ISO date YYYY-MM-DD"),
        "unit": (S, "'celsius' or 'fahrenheit'")},
       ["city", "date"], "天气 预报 下雨 降水 气温 rain temperature"),
    _t("get_air_quality",
       "Get the current air quality index (AQI) and main pollutants for a city.",
       {"city": (S, "City name")}, ["city"], "空气质量 污染 pm2.5"),
    # ---- 邮件 / 通讯 ----
    _t("send_email",
       "Send an email to one or more recipients. Supports plain-text and HTML bodies and "
       "optional CC/BCC lists.",
       {"to": (S, "Recipient email address"), "subject": (S, "Subject line"),
        "body": (S, "Email body"), "cc": (S, "Comma separated CC addresses")},
       ["to", "subject", "body"], "邮件 发送邮件 通知 提醒 mail notify"),
    _t("search_email",
       "Search the user's mailbox by sender, subject keywords or date range.",
       {"query": (S, "Search query"), "limit": (N, "Max results")}, ["query"], "邮箱 搜索邮件 收件箱 inbox"),
    _t("slack_post_message",
       "Post a message to a Slack channel or direct message thread.",
       {"channel": (S, "Channel name such as #dev"), "text": (S, "Message text")},
       ["channel", "text"], "slack 频道 群消息 通知团队 chat"),
    _t("slack_search_messages",
       "Search message history across Slack channels.",
       {"query": (S, "Search query")}, ["query"], "slack 历史消息 搜索"),
    _t("send_sms",
       "Send an SMS text message to a phone number.",
       {"phone": (S, "E.164 phone number"), "text": (S, "Message text")},
       ["phone", "text"], "短信 手机 text message"),
    # ---- 日历 ----
    _t("calendar_create_event",
       "Create a calendar event with title, start/end time, attendees and optional location.",
       {"title": (S, "Event title"), "start": (S, "ISO datetime"), "end": (S, "ISO datetime"),
        "attendees": (S, "Comma separated emails")},
       ["title", "start", "end"], "日历 日程 会议 安排 schedule meeting"),
    _t("calendar_list_events",
       "List calendar events in a date range.",
       {"from_date": (S, "Start date"), "to_date": (S, "End date")},
       ["from_date", "to_date"], "日历 查看日程 agenda"),
    _t("calendar_find_free_slot",
       "Find a free time slot that works for a list of attendees.",
       {"attendees": (S, "Comma separated emails"), "duration_min": (N, "Duration in minutes")},
       ["attendees", "duration_min"], "空闲时间 约会 协调 availability"),
    # ---- GitHub / 代码 ----
    _t("github_list_issues",
       "List issues of a GitHub repository, filterable by state, label and assignee.",
       {"repo": (S, "owner/name"), "state": (S, "open | closed | all"), "label": (S, "Label filter")},
       ["repo"], "github 仓库 issue 问题 工单"),
    _t("github_create_issue",
       "Create a new issue in a GitHub repository.",
       {"repo": (S, "owner/name"), "title": (S, "Issue title"), "body": (S, "Issue body")},
       ["repo", "title"], "github 新建 issue 提 bug"),
    _t("github_get_pull_request",
       "Get details, diff stats and review status of a pull request.",
       {"repo": (S, "owner/name"), "number": (N, "PR number")}, ["repo", "number"], "github pr 合并请求"),
    _t("github_merge_pull_request",
       "Merge a pull request using merge, squash or rebase strategy.",
       {"repo": (S, "owner/name"), "number": (N, "PR number"), "method": (S, "merge|squash|rebase")},
       ["repo", "number"], "github 合并 pr"),
    _t("github_search_code",
       "Search code across GitHub repositories.",
       {"query": (S, "Code search query")}, ["query"], "github 代码搜索"),
    # ---- 数据库 ----
    _t("sql_query",
       "Run a read-only SQL query against the analytics warehouse and return rows as JSON.",
       {"sql": (S, "SELECT statement"), "limit": (N, "Row limit")}, ["sql"], "数据库 查询 sql 报表 database"),
    _t("sql_describe_table",
       "Describe the columns and types of a warehouse table.",
       {"table": (S, "Table name")}, ["table"], "表结构 schema 字段"),
    _t("redis_get",
       "Read a key from the Redis cache.",
       {"key": (S, "Redis key")}, ["key"], "缓存 redis"),
    # ---- 文件 ----
    _t("read_file",
       "Read the text contents of a file from the workspace.",
       {"path": (S, "File path")}, ["path"], "读取文件 打开文件"),
    _t("write_file",
       "Write text content to a file in the workspace, creating it if necessary.",
       {"path": (S, "File path"), "content": (S, "Content")}, ["path", "content"], "写文件 保存 创建文件"),
    _t("list_directory",
       "List files and folders in a directory.",
       {"path": (S, "Directory path")}, ["path"], "目录 文件列表 ls"),
    _t("convert_document",
       "Convert a document between formats such as docx, pdf, markdown and html.",
       {"path": (S, "Input file"), "to": (S, "Target format")}, ["path", "to"], "转换 格式 pdf word"),
    # ---- 金融 / 支付 ----
    _t("currency_convert",
       "Convert an amount between two currencies using the latest exchange rate.",
       {"amount": (N, "Amount"), "from_ccy": (S, "Source currency code"), "to_ccy": (S, "Target currency code")},
       ["amount", "from_ccy", "to_ccy"], "汇率 换算 货币 美元 日元 exchange rate"),
    _t("stock_quote",
       "Get the latest quote and daily change for a stock ticker.",
       {"ticker": (S, "Ticker symbol")}, ["ticker"], "股票 股价 行情"),
    _t("create_invoice",
       "Create and send an invoice to a customer.",
       {"customer_id": (S, "Customer id"), "amount": (N, "Amount"), "currency": (S, "Currency")},
       ["customer_id", "amount"], "发票 开票 账单 billing"),
    _t("refund_payment",
       "Refund a previously captured payment, fully or partially.",
       {"payment_id": (S, "Payment id"), "amount": (N, "Amount to refund")}, ["payment_id"], "退款 支付"),
    # ---- 工单 / 项目管理 ----
    _t("jira_create_ticket",
       "Create a Jira ticket in a project with summary, description and priority.",
       {"project": (S, "Project key"), "summary": (S, "Summary"), "priority": (S, "Priority")},
       ["project", "summary"], "jira 工单 任务 创建"),
    _t("jira_search",
       "Search Jira tickets using JQL.",
       {"jql": (S, "JQL query")}, ["jql"], "jira 查询 工单"),
    _t("notion_create_page",
       "Create a Notion page under a parent page or database.",
       {"parent": (S, "Parent id"), "title": (S, "Title"), "content": (S, "Markdown content")},
       ["parent", "title"], "notion 笔记 文档 页面"),
    _t("notion_search",
       "Search pages and databases in Notion.",
       {"query": (S, "Search query")}, ["query"], "notion 搜索"),
    # ---- 出行 / 生活 ----
    _t("search_flights",
       "Search flights between two airports on a date, with price and duration.",
       {"origin": (S, "IATA code"), "destination": (S, "IATA code"), "date": (S, "ISO date")},
       ["origin", "destination", "date"], "机票 航班 飞机 flight"),
    _t("book_hotel",
       "Book a hotel room for given dates.",
       {"hotel_id": (S, "Hotel id"), "check_in": (S, "Date"), "check_out": (S, "Date")},
       ["hotel_id", "check_in", "check_out"], "酒店 预订 住宿"),
    _t("search_restaurants",
       "Search restaurants by cuisine, location and price range.",
       {"location": (S, "Location"), "cuisine": (S, "Cuisine")}, ["location"], "餐厅 美食 吃饭"),
    _t("get_directions",
       "Get route directions and travel time between two places.",
       {"origin": (S, "Start"), "destination": (S, "End"), "mode": (S, "driving|transit|walking")},
       ["origin", "destination"], "导航 路线 交通"),
    # ---- 数据 / 分析 / 其它 ----
    _t("web_search",
       "Search the public web and return top results with snippets.",
       {"query": (S, "Search query")}, ["query"], "网页搜索 联网 google"),
    _t("translate_text",
       "Translate text between languages.",
       {"text": (S, "Text"), "target_lang": (S, "Target language code")}, ["text", "target_lang"], "翻译 语言"),
    _t("generate_chart",
       "Render a line, bar or pie chart from tabular data and return an image URL.",
       {"type": (S, "line|bar|pie"), "data": (S, "JSON data")}, ["type", "data"], "图表 可视化 画图"),
    _t("run_python",
       "Execute a Python snippet in a sandbox and return stdout.",
       {"code": (S, "Python code")}, ["code"], "运行代码 计算 脚本"),
    _t("text_to_speech",
       "Convert text into spoken audio.",
       {"text": (S, "Text"), "voice": (S, "Voice name")}, ["text"], "语音 朗读 tts"),
    _t("image_resize",
       "Resize or crop an image to given dimensions.",
       {"path": (S, "Image path"), "width": (N, "Width"), "height": (N, "Height")},
       ["path", "width", "height"], "图片 缩放 裁剪"),
    _t("set_reminder",
       "Create a personal reminder at a given time.",
       {"text": (S, "Reminder text"), "time": (S, "ISO datetime")}, ["text", "time"], "提醒 闹钟 待办"),
    _t("smart_home_set_light",
       "Turn a smart-home light on or off and set its brightness.",
       {"room": (S, "Room"), "on": (S, "true|false"), "brightness": (N, "0-100")}, ["room", "on"], "智能家居 灯 开关"),
    _t("hr_lookup_employee",
       "Look up an employee's department, manager and contact info.",
       {"name": (S, "Employee name")}, ["name"], "员工 通讯录 人事 同事"),
]


def execute_tool(name: str, args: dict) -> str:
    """模拟工具执行：返回一段看起来合理的 JSON 字符串。"""
    import json
    fake = {
        "get_weather_forecast": {"city": args.get("city"), "date": args.get("date"),
                                 "summary": "Rain", "precip_prob": 0.8, "temp_c": [14, 19]},
        "send_email": {"status": "sent", "to": args.get("to")},
        "github_list_issues": {"issues": [{"number": 42, "title": "Crash on startup"},
                                          {"number": 41, "title": "Docs typo"}]},
        "slack_post_message": {"ok": True, "channel": args.get("channel")},
        "currency_convert": {"result": round(float(args.get("amount", 0)) * 151.2, 2), "rate": 151.2},
    }
    return json.dumps(fake.get(name, {"status": "ok", "note": "simulated result"}), ensure_ascii=False)
