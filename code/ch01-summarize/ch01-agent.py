# -*- coding: utf-8 -*-
"""
AI Agent 实验：销售报表分析（本地 llama.cpp + Gemma，openai SDK 通信）

实验目的：
    证明 Agent 不是“把问题发给模型再显示答案”，而是一个循环：
    目标 -> 模型决策 -> 工具执行 -> 结果反馈 -> 状态更新 -> 判断是否停止

使用方法：
    1. 先启动 llama.cpp 服务，例如：
           llama-server -m gemma-4.gguf --port 8080
    2. 安装依赖：  pip install openai
    3. 运行：
           python agent_experiment.py              # 依次运行 4 个场景
           python agent_experiment.py normal       # 只运行某一个场景

可选环境变量：
    LLAMA_BASE_URL  默认 http://127.0.0.1:8080/v1
    LLAMA_MODEL     默认 gemma-4

安全说明：
    - 只有三个只读工具，只读取本地 CSV 文件。
    - 不连接真实账户，不付款、不删除、不发送。
    - 实验开始和结束时对原始文件做哈希对比，证明文件没有被修改。
"""

import csv
import hashlib
import json
import os
import sys

from openai import OpenAI

# ------------------------------------------------------------
# 一、配置
# ------------------------------------------------------------
BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.environ.get("LLAMA_MODEL", "gemma-4")

MAX_STEPS = 10                 # 最多循环多少步（硬上限）
MAX_CONSECUTIVE_FAILURES = 3   # 连续失败（报错/空结果/格式错误）多少次就停止
MAX_SAME_CALL = 3              # 完全相同的工具调用连续出现多少次就停止

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_lab_data")
DETAIL_FILE = os.path.join(DATA_DIR, "sales_detail.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "sales_summary.csv")

CURRENT_QUARTER = "2026Q3"
PREVIOUS_QUARTER = "2026Q2"

QUARTER_MONTHS = {
    "2026Q2": ["2026-04", "2026-05", "2026-06"],
    "2026Q3": ["2026-07", "2026-08", "2026-09"],
}

USER_TASK = (
    "请分析本季度各地区销售情况，找出下降最明显的地区，核对数据后给我一份简短报告。"
    "不要修改原始文件。（本季度是 2026Q3，上一季度是 2026Q2，金额单位：万元）"
)

# 四个实验场景
SCENARIOS = ["normal", "timeout_once", "empty_once", "always_timeout"]
SCENARIO_TEXT = {
    "normal": "场景1：工具全部正常返回",
    "timeout_once": "场景2：第 3 次工具调用模拟超时（只超时一次，之后恢复）",
    "empty_once": "场景3：第 3 次工具调用模拟返回空结果（只空一次，之后恢复）",
    "always_timeout": "场景4：明细查询工具一直超时（检验能否在上限处停止）",
}

# ------------------------------------------------------------
# 二、准备实验数据（这一步由实验程序完成，不是 Agent 做的）
# ------------------------------------------------------------
def prepare_data_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(DETAIL_FILE):
        rows = [
            ["region", "month", "amount"],
            # 华东：增长
            ["华东", "2026-04", 320], ["华东", "2026-05", 330], ["华东", "2026-06", 340],
            ["华东", "2026-07", 350], ["华东", "2026-08", 360], ["华东", "2026-09", 370],
            # 华南：小幅下降（Q2=870, Q3=830）
            ["华南", "2026-04", 280], ["华南", "2026-05", 290], ["华南", "2026-06", 300],
            ["华南", "2026-07", 270], ["华南", "2026-08", 275], ["华南", "2026-09", 285],
            # 华北：下降最明显（Q2=1230, Q3=1000）
            ["华北", "2026-04", 400], ["华北", "2026-05", 410], ["华北", "2026-06", 420],
            ["华北", "2026-07", 380], ["华北", "2026-08", 330], ["华北", "2026-09", 290],
            # 西南：基本持平略降（Q2=465, Q3=450）
            ["西南", "2026-04", 150], ["西南", "2026-05", 155], ["西南", "2026-06", 160],
            ["西南", "2026-07", 150], ["西南", "2026-08", 150], ["西南", "2026-09", 150],
        ]
        f = open(DETAIL_FILE, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()

    if not os.path.exists(SUMMARY_FILE):
        # 故意让“华南 2026Q3”的汇总值(840)与明细合计(830)不一致，用来观察“核对数据”
        rows = [
            ["region", "quarter", "total"],
            ["华东", "2026Q2", 990], ["华东", "2026Q3", 1080],
            ["华南", "2026Q2", 870], ["华南", "2026Q3", 840],
            ["华北", "2026Q2", 1230], ["华北", "2026Q3", 1000],
            ["西南", "2026Q2", 465], ["西南", "2026Q3", 450],
        ]
        f = open(SUMMARY_FILE, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()


def file_hash(path):
    f = open(path, "rb")
    content = f.read()
    f.close()
    return hashlib.md5(content).hexdigest()


def read_csv_rows(path):
    # 只读方式打开
    f = open(path, "r", encoding="utf-8")
    reader = csv.DictReader(f)
    rows = []
    for row in reader:
        rows.append(row)
    f.close()
    return rows


# ------------------------------------------------------------
# 三、三个只读工具
# ------------------------------------------------------------
def tool_list_regions(args):
    rows = read_csv_rows(SUMMARY_FILE)
    regions = []
    for row in rows:
        if row["region"] not in regions:
            regions.append(row["region"])
    return regions


def tool_query_detail(args):
    region = args.get("region", "")
    quarter = args.get("quarter", "")
    months = QUARTER_MONTHS.get(quarter, [])
    rows = read_csv_rows(DETAIL_FILE)
    result = []
    for row in rows:
        if row["region"] == region and row["month"] in months:
            result.append({"month": row["month"], "amount": int(row["amount"])})
    return result


def tool_get_summary(args):
    region = args.get("region", "")
    quarter = args.get("quarter", "")
    rows = read_csv_rows(SUMMARY_FILE)
    for row in rows:
        if row["region"] == region and row["quarter"] == quarter:
            return {"region": region, "quarter": quarter, "total": int(row["total"])}
    return []


TOOLS = {
    "list_regions": tool_list_regions,
    "query_detail": tool_query_detail,
    "get_summary": tool_get_summary,
}


def run_tool(name, args, scenario, call_number):
    """
    执行工具，并按场景注入故障。
    返回统一格式：{"ok": True/False, "data": ..., "error": "..."}
    """
    if name not in TOOLS:
        return {"ok": False, "data": None, "error": "未知工具: " + str(name)}

    # ---- 故障注入（模拟外部系统出问题）----
    if scenario == "timeout_once" and call_number == 3:
        return {"ok": False, "data": None, "error": "TimeoutError: 工具调用超时（模拟）"}
    if scenario == "empty_once" and call_number == 3:
        return {"ok": True, "data": [], "error": ""}
    if scenario == "always_timeout" and name == "query_detail":
        return {"ok": False, "data": None, "error": "TimeoutError: 工具调用超时（模拟）"}

    # ---- 正常执行 ----
    try:
        data = TOOLS[name](args)
        return {"ok": True, "data": data, "error": ""}
    except Exception as e:
        return {"ok": False, "data": None, "error": "工具执行异常: " + str(e)}


# ------------------------------------------------------------
# 四、提示词
# ------------------------------------------------------------
SYSTEM_PROMPT = """你是一个销售数据分析助手，只能通过工具获取数据，不能编造数字。
每次回复只能输出一个 JSON 对象，不要输出任何其他文字。

格式一：调用工具
{"thought": "一句话说明理由", "action": "call_tool", "tool": "工具名", "args": {参数}}

格式二：给出最终报告
{"thought": "一句话说明理由", "action": "final", "answer": "简短报告"}

可用工具（全部只读）：
1. list_regions    参数：{}                                  返回所有地区名称
2. query_detail    参数：{"region": "地区", "quarter": "2026Q3"}   返回该地区该季度各月销售额明细
3. get_summary     参数：{"region": "地区", "quarter": "2026Q3"}   返回该地区该季度的汇总销售额

工作规则：
- 需要比较 2026Q2 和 2026Q3 两个季度。
- “核对数据”的意思是：用 query_detail 明细合计，与 get_summary 汇总值对比，发现不一致要在报告里指出。
- 每次工具返回会带有 status：ok 表示成功，error 表示失败，empty 表示空结果。
- 工具失败或空结果时，不要编造数据。可以重试，或换一种方式取数。
- 如果多次仍然拿不到数据，就在最终报告里如实说明哪些数据缺失、无法下结论。
- 数据足够后，再给出最终报告。"""


# ------------------------------------------------------------
# 五、辅助函数
# ------------------------------------------------------------
def print_line(char="="):
    print(char * 70)


def parse_action(text):
    """从模型回复里取出 JSON。返回 (动作字典, 错误说明)。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "回复中没有找到 JSON 对象"
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError as e:
        return None, "JSON 解析失败: " + str(e)
    if not isinstance(obj, dict):
        return None, "JSON 不是对象"
    action = obj.get("action", "")
    if action != "call_tool" and action != "final":
        return None, "action 必须是 call_tool 或 final"
    return obj, ""


def new_state(scenario):
    state = {
        "scenario": scenario,
        "step": 0,                    # 已执行的循环步数
        "tool_calls": 0,              # 已执行的工具调用次数
        "consecutive_failures": 0,    # 连续失败次数
        "total_failures": 0,          # 总失败次数
        "regions": [],                # 已知地区
        "detail_totals": {},          # "地区|季度" -> 明细合计
        "summary_totals": {},         # "地区|季度" -> 汇总值
        "last_call_key": "",          # 上一次工具调用的“指纹”
        "same_call_count": 0,         # 同一指纹连续出现次数
        "stop_reason": "",
        "final_answer": "",
    }
    return state


def show_state(state):
    print("  步数=" + str(state["step"]) + "/" + str(MAX_STEPS)
          + "  工具调用次数=" + str(state["tool_calls"])
          + "  连续失败=" + str(state["consecutive_failures"]) + "/" + str(MAX_CONSECUTIVE_FAILURES)
          + "  总失败=" + str(state["total_failures"]))
    print("  已知地区: " + json.dumps(state["regions"], ensure_ascii=False))
    print("  已取得的明细合计: " + json.dumps(state["detail_totals"], ensure_ascii=False))
    print("  已取得的汇总值:   " + json.dumps(state["summary_totals"], ensure_ascii=False))


def judge_status(result):
    """把工具返回归类为 ok / error / empty。"""
    if not result["ok"]:
        return "error"
    data = result["data"]
    if data is None or len(data) == 0:
        return "empty"
    return "ok"


def update_state_after_tool(state, name, args, result, status):
    """根据工具真实返回值更新状态。只有成功且非空的数据才会被记录。"""
    state["tool_calls"] = state["tool_calls"] + 1

    if status == "ok":
        state["consecutive_failures"] = 0
        if name == "list_regions":
            state["regions"] = result["data"]
        if name == "query_detail":
            total = 0
            for item in result["data"]:
                total = total + item["amount"]
            key = args.get("region", "") + "|" + args.get("quarter", "")
            state["detail_totals"][key] = total
        if name == "get_summary":
            key = args.get("region", "") + "|" + args.get("quarter", "")
            state["summary_totals"][key] = result["data"]["total"]
    else:
        state["consecutive_failures"] = state["consecutive_failures"] + 1
        state["total_failures"] = state["total_failures"] + 1


def check_final_answer(state):
    """
    用 Agent 实际取得的数据，自己算一遍，再和模型的最终回答对比。
    """
    print("  【一致性检查】用工具真实返回的数据重新计算：")
    best_region = ""
    best_change = 0.0
    complete_regions = 0

    for region in state["regions"]:
        cur_key = region + "|" + CURRENT_QUARTER
        pre_key = region + "|" + PREVIOUS_QUARTER
        if cur_key in state["detail_totals"] and pre_key in state["detail_totals"]:
            cur = state["detail_totals"][cur_key]
            pre = state["detail_totals"][pre_key]
            change = (cur - pre) * 100.0 / pre
            complete_regions = complete_regions + 1
            print("    " + region + ": " + str(pre) + " -> " + str(cur)
                  + "  变化 " + str(round(change, 1)) + "%")
            if change < best_change:
                best_change = change
                best_region = region
        else:
            print("    " + region + ": 数据不完整，Agent 并没有取到两个季度的明细")

    # 汇总值与明细合计的核对
    mismatch_list = []
    for region in state["regions"]:
        for quarter in [PREVIOUS_QUARTER, CURRENT_QUARTER]:
            key = region + "|" + quarter
            if key in state["detail_totals"] and key in state["summary_totals"]:
                if state["detail_totals"][key] != state["summary_totals"][key]:
                    mismatch_list.append(key + "（明细 " + str(state["detail_totals"][key])
                                         + "，汇总 " + str(state["summary_totals"][key]) + "）")
    if len(mismatch_list) > 0:
        print("    Agent 取得的数据里存在核对不一致: " + "; ".join(mismatch_list))
    else:
        print("    Agent 取得的数据里没有发现核对不一致（可能没有取到足够数据）")

    answer = state["final_answer"]
    if state["stop_reason"] != "final_answer":
        print("    没有模型给出的最终答案（因为被上限或失败条件中止）。")
        return

    if best_region == "":
        print("    数据不完整，无法算出下降最明显地区。")
        if "无法" in answer or "缺失" in answer or "未能" in answer or "没有" in answer:
            print("    -> 模型的报告承认了数据不足：一致。")
        else:
            print("    -> 模型却给出了确定结论：【不一致】，可能是编造的。")
        return

    print("    程序算出的下降最明显地区: " + best_region + "（" + str(round(best_change, 1)) + "%）")
    if complete_regions < len(state["regions"]):
        print("    注意：只有部分地区数据完整，结论可能不可靠。")
    if best_region in answer:
        print("    -> 模型回答中提到了该地区：一致。")
    else:
        print("    -> 模型回答中没有提到该地区：【不一致】。")

    if len(mismatch_list) > 0:
        if "不一致" in answer or "差异" in answer or "不符" in answer:
            print("    -> 模型报告指出了数据差异：一致。")
        else:
            print("    -> 模型报告没有指出数据差异：【遗漏】。")


# ------------------------------------------------------------
# 六、Agent 主循环
# ------------------------------------------------------------
def run_agent(client, scenario):
    print_line("=")
    print(SCENARIO_TEXT[scenario])
    print_line("=")
    print("用户任务: " + USER_TASK)
    print("停止条件: 模型给出最终答案 / 达到 " + str(MAX_STEPS) + " 步 / 连续失败 "
          + str(MAX_CONSECUTIVE_FAILURES) + " 次 / 同一调用重复 " + str(MAX_SAME_CALL) + " 次")

    state = new_state(scenario)
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": USER_TASK})

    while True:
        # ---- 停止条件检查（在每一步开始前）----
        if state["step"] >= MAX_STEPS:
            state["stop_reason"] = "max_steps（达到最大步数）"
            break

        state["step"] = state["step"] + 1
        print("")
        print_line("-")
        print("第 " + str(state["step"]) + " 步")
        print_line("-")

        # ---- 1. 输入：这一步发给模型的最新一条消息 ----
        last_message = messages[len(messages) - 1]
        print("[1] 输入给模型的最新消息 (role=" + last_message["role"] + "):")
        print("    " + last_message["content"])
        print("    （模型每次都能看到此前全部 " + str(len(messages)) + " 条消息，这就是它的“记忆”）")

        # ---- 2. 模型决策 ----
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=800,
            )
            reply = response.choices[0].message.content
            if reply is None:
                reply = ""
        except Exception as e:
            print("[2] 调用模型失败: " + str(e))
            state["stop_reason"] = "model_error（模型服务出错）"
            break

        print("[2] 模型原始输出:")
        print("    " + reply.replace("\n", "\n    "))
        messages.append({"role": "assistant", "content": reply})

        action, error_text = parse_action(reply)

        # ---- 格式错误：算一次失败，并告诉模型 ----
        if action is None:
            state["consecutive_failures"] = state["consecutive_failures"] + 1
            state["total_failures"] = state["total_failures"] + 1
            print("[3] 解析模型行动失败: " + error_text)
            feedback = "你的回复格式不对（" + error_text + "）。请只输出一个符合格式的 JSON 对象。"
            messages.append({"role": "user", "content": feedback})
            print("[4] 状态更新:")
            show_state(state)
            if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                state["stop_reason"] = "too_many_failures（连续失败过多）"
                break
            continue

        print("[3] 模型提出的行动: " + json.dumps(action, ensure_ascii=False))

        # ---- 最终答案：停止 ----
        if action["action"] == "final":
            state["final_answer"] = str(action.get("answer", ""))
            state["stop_reason"] = "final_answer（模型给出最终答案）"
            print("[4] 模型选择结束任务，不再调用工具。")
            show_state(state)
            break

        # ---- 调用工具 ----
        tool_name = str(action.get("tool", ""))
        tool_args = action.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        # 重复调用检测（防止原地打转）
        call_key = tool_name + json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
        if call_key == state["last_call_key"]:
            state["same_call_count"] = state["same_call_count"] + 1
        else:
            state["same_call_count"] = 1
        state["last_call_key"] = call_key

        if state["same_call_count"] >= MAX_SAME_CALL:
            print("[4] 同一个调用已连续出现 " + str(state["same_call_count"]) + " 次，程序判定为原地打转。")
            state["stop_reason"] = "repeated_action（重复同一行动）"
            break

        call_number = state["tool_calls"] + 1
        result = run_tool(tool_name, tool_args, scenario, call_number)
        status = judge_status(result)
        print("[4] 工具实际返回 (第 " + str(call_number) + " 次调用, status=" + status + "):")
        print("    " + json.dumps(result, ensure_ascii=False))

        # ---- 5. 更新状态 ----
        update_state_after_tool(state, tool_name, tool_args, result, status)
        print("[5] 状态更新:")
        show_state(state)

        # ---- 6. 把结果反馈给模型 ----
        observation = {"tool": tool_name, "status": status, "result": result}
        observation_text = "工具返回: " + json.dumps(observation, ensure_ascii=False)
        messages.append({"role": "user", "content": observation_text})

        if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            print("    连续失败已达上限，程序停止循环。")
            state["stop_reason"] = "too_many_failures（连续失败过多）"
            break

    # ---- 循环结束，输出结果 ----
    print("")
    print_line("-")
    print("本场景结束")
    print_line("-")
    print("停止原因: " + state["stop_reason"])
    if state["final_answer"] != "":
        print("模型的最终答案:")
        print("    " + state["final_answer"].replace("\n", "\n    "))
    else:
        print("模型没有给出最终答案。程序的兜底说明：任务未完成，已取得的数据如下。")
        show_state(state)
    check_final_answer(state)
    return state


# ------------------------------------------------------------
# 七、主程序
# ------------------------------------------------------------
def main():
    prepare_data_files()

    hash_before_detail = file_hash(DETAIL_FILE)
    hash_before_summary = file_hash(SUMMARY_FILE)
    print("原始文件哈希（实验前）:")
    print("  sales_detail.csv  " + hash_before_detail)
    print("  sales_summary.csv " + hash_before_summary)

    client = OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=120)
    print("模型服务: " + BASE_URL + "  模型名: " + MODEL_NAME)

    # 选择场景
    run_list = SCENARIOS
    if len(sys.argv) > 1:
        if sys.argv[1] in SCENARIOS:
            run_list = [sys.argv[1]]
        else:
            print("未知场景: " + sys.argv[1] + "，可选: " + ", ".join(SCENARIOS))
            return

    results = []
    for scenario in run_list:
        state = run_agent(client, scenario)
        results.append(state)
        print("")

    # 汇总对比
    print_line("=")
    print("实验汇总")
    print_line("=")
    for state in results:
        print(state["scenario"] + ": 步数=" + str(state["step"])
              + ", 工具调用=" + str(state["tool_calls"])
              + ", 失败=" + str(state["total_failures"])
              + ", 停止原因=" + state["stop_reason"])

    hash_after_detail = file_hash(DETAIL_FILE)
    hash_after_summary = file_hash(SUMMARY_FILE)
    print("")
    if hash_before_detail == hash_after_detail and hash_before_summary == hash_after_summary:
        print("原始文件哈希未变化：Agent 没有修改原始文件。")
    else:
        print("警告：原始文件哈希发生变化！")

    print("")
    print("观察提示：")
    print("  1. 对比每一步的 [3] 模型提出的行动 和 [4] 工具实际返回，模型只“提议”，工具由程序执行。")
    print("  2. 看 [5] 状态：只有成功且非空的数据才进入状态，失败和空结果不会被当成数据。")
    print("  3. 场景2/3：看模型在失败/空结果后，是否重试或换方式，最终答案是否仍然正确。")
    print("  4. 场景4：看程序是否在连续失败上限处停止，最终说明是否如实承认数据缺失。")


if __name__ == "__main__":
    main()
