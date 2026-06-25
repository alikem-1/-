from datetime import datetime

def get_current_week():
    """获取当前校历周数（假设2026年3月2日开学）"""
    today = datetime.now()
    start_date = datetime(2026, 3, 2)   # 固定开学日期
    delta = today - start_date
    week_num = delta.days // 7 + 1
    if week_num < 1:
        week_num = 1
    return f"📅 现在是第 {week_num} 周（校历）"

def calculate_gpa(score_str: str):
    """百分制成绩转标准4.0绩点，输出详细明细"""
    try:
        score_list = [s.strip() for s in score_str.split(",")]
        scores = []
        for s in score_list:
            if not s.isdigit():
                return f"❌ 输入错误：「{s}」不是合法数字，请重新输入！"
            num = float(s)
            if not 0 <= num <= 100:
                return f"❌ 分数「{num}」超出0-100范围，无效！"
            scores.append(num)

        # 高校通用绩点换算标准
        def score2gpa(s):
            if s >= 90: return 4.0
            elif s >= 85: return 3.7
            elif s >= 82: return 3.3
            elif s >= 78: return 3.0
            elif s >= 75: return 2.7
            elif s >= 72: return 2.3
            elif s >= 68: return 2.0
            elif s >= 64: return 1.5
            elif s >= 60: return 1.0
            else: return 0.0

        gpa_list = [score2gpa(i) for i in scores]
        avg_score = round(sum(scores)/len(scores), 2)
        avg_gpa = round(sum(gpa_list)/len(gpa_list), 2)

        # 格式化输出明细
        output = "### 📊 GPA计算结果\n"
        output += f"- 总科目数：{len(scores)} 门\n"
        output += f"- 平均分：{avg_score}\n"
        output += f"- 平均绩点：{avg_gpa}\n\n"
        output += "#### 各科明细\n"
        for s, g in zip(scores, gpa_list):
            output += f"{s} 分 → {g} 绩点\n"
        return output

    except Exception as e:
        return f"❌ 计算失败：{str(e)}，格式示例：85,90,78"
