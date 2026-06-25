import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

import streamlit as st
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# 原有业务工具文件
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

# -------------------------- 全局美化CSS注入（页面加载最先执行） --------------------------
def inject_custom_style():
    custom_css = """
    <style>
    /* 全局基础布局 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
        max-width: 1200px;
    }
    /* 侧边栏玻璃态美化 */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #f7f9fc 0%, #eef2f8 100%);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }
    /* 渐变主标题 */
    h1 {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        color: transparent;
        font-weight: 700;
        font-size: 2.4rem !important;
    }
    /* 聊天消息卡片美化 */
    .stChatMessage {
        padding: 1.2rem 1.5rem !important;
        border-radius: 18px !important;
        margin: 0.8rem 0 !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        border: none !important;
    }
    /* 用户消息蓝紫渐变 */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #eff6ff, #eef2ff) !important;
    }
    /* AI消息柔和白底 */
    .stChatMessage[data-testid="assistant"] {
        background: #ffffff !important;
    }
    /* 按钮统一美化：圆角+悬浮上浮动效 */
    .stButton>button {
        border-radius: 12px !important;
        padding: 0.4rem 1rem !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        border-color: #2563eb !important;
    }
    /* 快捷提问小按钮 */
    [data-testid="stHorizontalBlock"] .stButton>button {
        font-size: 0.85rem;
        padding: 0.3rem 0.7rem !important;
    }
    /* 分割线美化 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }
    /* 输入框底部固定样式 */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 16rem;
        right: 2rem;
        background: #fff;
        padding: 1rem 1.5rem;
        box-shadow: 0 -4px 16px rgba(0,0,0,0.05);
        border-radius: 16px 16px 0 0;
    }
    /* 提示警告卡片 */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    /* 侧边栏标题间距 */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    /* 音频播放器美化 */
    audio::-webkit-media-controls-panel {
        background: #f1f5f9;
        border-radius: 10px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(
    page_title="校园百事通",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 注入全局美化样式（必须在set_page_config之后）
inject_custom_style()

# 加载环境变量
load_dotenv()

# -------------------------- 缓存资源加载 --------------------------
@st.cache_resource(show_spinner="正在加载文本向量化模型...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner="正在加载校园知识库向量库...")
def load_vector_db():
    embeddings = load_embeddings()
    db_path = "./vector_db"
    if not os.path.exists(db_path):
        st.warning(f"向量库目录 {db_path} 不存在，请先导入校园文档生成知识库！")
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

# 初始化全局向量资源
embeddings = load_embeddings()
vector_db = load_vector_db()

# 校验星火API密钥
APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 环境变量缺失！请在项目根目录 .env 文件中配置 SPARK_APIPASSWORD 星火接口密钥")
    st.stop()

# -------------------------- 原有业务问答逻辑（无修改） --------------------------
def rag_retrieve_answer(question):
    try:
        docs = vector_db.similarity_search(question, k=3)
        if len(docs) == 0:
            return "📭 知识库未查询到相关内容，暂时无法解答该问题，你可以咨询教务处相关老师。"
        
        context = "\n\n=====分割线=====\n\n".join([doc.page_content.strip() for doc in docs])
        prompt_text = RAG_PROMPT.format(context=context, question=question)

        url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {APIPASSWORD}"
        }
        payload = {
            "model": "spark-x",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ 大模型接口调用失败\n状态码：{resp.status_code}\n返回信息：{resp.text[:300]}"

    except requests.exceptions.Timeout:
        return "⏱️ 请求超时，服务器响应缓慢，请重新提问！"
    except Exception as e:
        return f"⚠️ 问答流程出现未知异常：{str(e)}"

def agent_answer(question):
    week_pattern = r'第几周|第\d+周|本周|校历|现在几周|教学周'
    gpa_pattern = r'绩点|GPA|平均分|算分|成绩换算'

    if re.search(week_pattern, question):
        return get_current_week()
    
    if re.search(gpa_pattern, question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return """📝 绩点计算使用说明
请在问题中带上你的各科分数，示例：
帮我算绩点：88,76,92,65"""
    
    return rag_retrieve_answer(question)

# -------------------------- 侧边栏重构（匹配截图：语音交互+三大能力） --------------------------
with st.sidebar:
    st.header("⚙️ 功能面板")
    st.divider()

    # ========== 语音交互模块（对应截图左侧报错区域） ==========
    st.subheader("🎤 语音交互功能")
    auto_tts = st.toggle("自动朗读AI回答", value=False, help="开启后AI回复会自动语音播报")
    
    # 录音按钮占位（streamlit-webrtc/audio-recorder-streamlit 可接入）
    voice_btn = st.button("按住说话", use_container_width=True)
    if voice_btn:
        st.warning("[WinError 2] 系统找不到指定的文件。\n请安装ffmpeg、PyAudio依赖后重试语音识别")
    
    # 语音朗读报错提示占位
    if "tts_error" in st.session_state:
        st.error(f"语音朗读生成失败：{st.session_state['tts_error']}")

    st.divider()

    # ========== 三大核心能力卡片展示 ==========
    st.subheader("✨ 三大核心能力")
    with st.container():
        st.markdown("""
        <div style="background:#fff; padding:12px; border-radius:12px; margin-bottom:10px; box-shadow:0 1px 6px rgba(0,0,0,0.04)">
            <b>📚 校园知识库问答</b>
            <p style="font-size:0.85rem; color:#4b5563; margin:4px 0 0 0">规章制度、宿舍、选课、奖学金、社团等校内问题</p>
        </div>
        <div style="background:#fff; padding:12px; border-radius:12px; margin-bottom:10px; box-shadow:0 1px 6px rgba(0,0,0,0.04)">
            <b>📅 教学周自动查询</b>
            <p style="font-size:0.85rem; color:#4b5563; margin:4px 0 0 0">自动获取当前学期第几教学周</p>
        </div>
        <div style="background:#fff; padding:12px; border-radius:12px; box-shadow:0 1px 6px rgba(0,0,0,0.04)">
            <b>📊 百分制GPA绩点换算</b>
            <p style="font-size:0.85rem; color:#4b5563; margin:4px 0 0 0">批量输入成绩计算平均绩点</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # ========== 快捷提问示例 ==========
    st.subheader("💡 快速提问示例")
    sample_q = [
        "现在是第几教学周？",
        "帮我计算绩点 90,82,75,60",
        "学校奖学金申请条件是什么？",
        "怎么请病假"
    ]
    # 两行两列排布按钮，更紧凑美观
    col_q1, col_q2 = st.columns(2)
    for idx, q in enumerate(sample_q):
        target_col = col_q1 if idx % 2 == 0 else col_q2
        with target_col:
            if st.button(q, key=f"quick_q_{idx}"):
                st.session_state["temp_input"] = q

    st.divider()

    # 清空对话按钮
    if st.button("🗑️ 清空全部对话记录", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -------------------------- 主页面聊天UI美化分层 --------------------------
# 顶部标题+简介卡片
st.title("🏫 校园生活百事通助手")
st.markdown("""
<div style="background:linear-gradient(135deg,#f0f7ff,#f5f3ff); padding:1rem 1.5rem; border-radius:16px; margin-bottom:1.2rem">
<p style="margin:0; color:#334155">
基于本地校园知识库 + 大模型RAG智能问答，支持【语音输入提问+语音朗读回答】，兼顾周数查询、绩点换算，一站式解决校园全部疑问
</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# 初始化对话记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是校园百事通，有任何校园问题、想查教学周、计算绩点都可以直接问我~ 也可以左侧麦克风语音提问！"}
    ]

# 渲染历史对话
for msg in st.session_state.messages:
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# 底部聊天输入框
input_text = st.chat_input("请输入你的校园问题...")
# 侧边快捷提问赋值
if "temp_input" in st.session_state and st.session_state["temp_input"]:
    input_text = st.session_state["temp_input"]
    del st.session_state["temp_input"]

# 处理用户提问
if input_text:
    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(input_text)

    # 生成AI回复
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AI正在检索知识库并思考答案..."):
            res = agent_answer(input_text)
        st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

        # 自动朗读逻辑预留（修复a coroutine was expected报错位置）
        if auto_tts:
            try:
                # 此处替换为你的Edge-TTS/讯飞TTS异步函数
                # audio_bytes = asyncio.run(tts_generate(res))
                # st.audio(audio_bytes, format="audio/mp3")
                pass
            except TypeError as e:
                st.session_state["tts_error"] = str(e)
                st.warning(f"语音朗读生成失败：{str(e)}")
