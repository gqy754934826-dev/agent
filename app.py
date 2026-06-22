import streamlit as st
import re
from agent.react_agent import XiaoQiAgent
from utils.memory import (
    create_session, list_sessions, get_session,
    get_history, add_message, delete_session,
)

st.set_page_config(
    page_title="小祺智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 样式 ----
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1e1e2e;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        color: #cdd6f4;
    }
    .role-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .role-code { background: #a6e3a1; color: #1e1e2e; }
    .role-psychology { background: #f5c2e7; color: #1e1e2e; }
    .role-history { background: #f9e2af; color: #1e1e2e; }
    .role-literature { background: #cba6f7; color: #1e1e2e; }
    .role-science { background: #89b4fa; color: #1e1e2e; }
</style>
""", unsafe_allow_html=True)


def strip_knowledge_tags(text: str) -> str:
    """清理回答中的知识库检索标记和思考过渡语"""
    # 清理知识库标记
    text = re.sub(r'\[知识库检索结果\].*?\[/知识库检索结果\]', '', text, flags=re.DOTALL).strip()
    # 清理常见的思考过渡语
    patterns = [
        r'^(让我|我来|我先|让我先|首先我|现在我).{0,20}(检索|查询|查找|搜索|看看|了解一下).{0,10}[，。,.\n]',
        r'^(我需要|我应该|我需要先).{0,20}(检索|查询|查找|搜索).{0,10}[，。,.\n]',
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.MULTILINE).strip()
    return text


ROLE_CONFIG = {
    "code":       {"label": "💻 代码专家",   "css": "role-code"},
    "psychology": {"label": "🧠 心理专家",   "css": "role-psychology"},
    "history":    {"label": "📜 历史专家",   "css": "role-history"},
    "literature": {"label": "📖 文学专家",   "css": "role-literature"},
    "science":    {"label": "🔬 科学专家",   "css": "role-science"},
    "base":       {"label": "🤖 通用助手",   "css": ""},
}


# ---- 初始化 Agent ----
@st.cache_resource
def get_agent():
    # 启动时自动加载 data/ 目录下的知识库到向量库
    with st.spinner("正在加载知识库..."):
        try:
            from rag.vector_store import VectorStoreService
            vs = VectorStoreService()
            vs.load_document()
        except Exception as e:
            st.warning(f"知识库加载: {e}")
    return XiaoQiAgent()

agent = get_agent()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def switch_session(session_id):
    st.session_state.current_session_id = session_id
    if session_id:
        history = get_history(session_id)
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"], "role_detected": m.get("role_detected", "base")}
            for m in history
        ]
    else:
        st.session_state.messages = []


# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("## 🤖 小祺助手")
    st.markdown("---")

    if st.button("＋ 新建对话", use_container_width=True, type="primary"):
        session = create_session()
        switch_session(session["id"])
        st.rerun()

    st.markdown("---")
    with st.expander("📂 历史对话", expanded=True):
        sessions = list_sessions()
        if not sessions:
            st.caption("暂无对话记录")
        else:
            for sess in sessions:
                col1, col2 = st.columns([5, 1])
                with col1:
                    is_active = sess["id"] == st.session_state.current_session_id
                    if st.button(
                        f"{'▸ ' if is_active else ''}{sess['title']}",
                        key=f"s_{sess['id']}",
                        use_container_width=True,
                    ):
                        switch_session(sess["id"])
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"d_{sess['id']}"):
                        delete_session(sess["id"])
                        if sess["id"] == st.session_state.current_session_id:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()

    st.markdown("---")

    # 文件上传
    st.markdown("### 📎 上传资料")
    uploaded_file = st.file_uploader(
        "支持 txt / pdf / md", type=["txt", "pdf", "md"], label_visibility="collapsed",
    )
    if uploaded_file:
        save_path = f"data/{uploaded_file.name}"
        temp_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        # MD5去重检查
        import hashlib, os
        from utils.path_tool import get_abs_path
        from utils.config_handler import chroma_conf
        md5 = hashlib.md5(temp_bytes).hexdigest()
        md5_file = get_abs_path(chroma_conf["md5_hex_store"])
        already_loaded = False
        if os.path.exists(md5_file):
            with open(md5_file, "r", encoding="utf-8") as f:
                if md5 in f.read():
                    already_loaded = True

        if already_loaded:
            st.info(f"ℹ️ {uploaded_file.name} 已在知识库中，无需重复上传")
        else:
            with open(save_path, "wb") as f:
                f.write(temp_bytes)
            st.success(f"✓ {uploaded_file.name} 已保存")
            with st.spinner("正在加载到知识库..."):
                try:
                    from rag.vector_store import VectorStoreService
                    from utils.file_handler import txt_loader, pdf_loader
                    vs = VectorStoreService()
                    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
                    if suffix in ("txt", "md"):
                        docs = txt_loader(save_path)
                    elif suffix == "pdf":
                        docs = pdf_loader(save_path)
                    else:
                        docs = []
                    split_docs = vs.spliter.split_documents(docs)
                    if split_docs:
                        vs.vector_store.add_documents(split_docs)
                        with open(md5_file, "a", encoding="utf-8") as f:
                            f.write(md5 + "\n")
                        st.success(f"✓ 已加载 {len(split_docs)} 个片段到知识库")
                    else:
                        st.warning("文件内容为空")
                except Exception as e:
                    st.error(f"加载失败: {e}")

    st.markdown("---")
    st.caption("基于 ReAct Agent + RAG 构建")


# ---- 主聊天区标题 ----
if st.session_state.current_session_id:
    sess = get_session(st.session_state.current_session_id)
    st.markdown(f"### {sess['title'] if sess else '新对话'}")
else:
    st.markdown("### 🤖 你好，我是小祺")


# ---- 渲染历史消息 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        role_detected = msg.get("role_detected", "base")
        if msg["role"] == "assistant" and role_detected and role_detected != "base":
            cfg = ROLE_CONFIG.get(role_detected, ROLE_CONFIG["base"])
            if cfg["css"]:
                st.markdown(
                    f'<span class="role-tag {cfg["css"]}">{cfg["label"]}</span>',
                    unsafe_allow_html=True,
                )
        st.markdown(msg["content"])


# ---- 用户输入 ----
if prompt := st.chat_input("输入你的问题..."):

    # 写入用户消息
    st.session_state.messages.append({"role": "user", "content": prompt, "role_detected": "base"})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 如果没有会话，先创建
    if not st.session_state.current_session_id:
        session = create_session()
        st.session_state.current_session_id = session["id"]

    session_id = st.session_state.current_session_id

    # 先保存用户消息
    add_message(session_id, "user", prompt)

    # 读取历史传给Agent（Agent通过messages列表记住对话）
    history = get_history(session_id, limit=20)
    history_for_agent = [{"role": m["role"], "content": m["content"]} for m in history]

    # ---- 调用Agent ----
    with st.chat_message("assistant"):
        # 思考动画状态
        status = st.status("🤔 小祺正在思考...", expanded=True)
        message_placeholder = st.empty()
        full_response = ""
        first_token = True
        actual_role = "base"

        try:
            for token in agent.execute_stream(
                query=prompt,
                session_id=session_id,
                history=history_for_agent,
            ):
                # 处理agent返回的角色标记
                if token.startswith("__role__:"):
                    actual_role = token.split(":", 1)[1]
                    cfg = ROLE_CONFIG.get(actual_role, ROLE_CONFIG["base"])
                    if cfg["css"]:
                        status.update(label=f"🔄 已切换到{cfg['label']}，正在生成回答...", state="running")
                    continue

                if first_token:
                    status.update(label="💡 正在生成回答...", state="running")
                    first_token = False
                full_response += token
                message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"抱歉，发生了错误：{str(e)}"
            st.error(full_response)

        if full_response:
            # 显示实际使用的角色标签
            if actual_role != "base":
                cfg = ROLE_CONFIG.get(actual_role, ROLE_CONFIG["base"])
                if cfg["css"]:
                    st.markdown(
                        f'<span class="role-tag {cfg["css"]}">{cfg["label"]}</span>',
                        unsafe_allow_html=True,
                    )
            message_placeholder.markdown(full_response)
            status.update(label="✅ 回答完成", state="complete")
        else:
            status.update(label="❌ 未获取到回答", state="error")

    # 清理标记后保存
    clean_response = strip_knowledge_tags(full_response)
    # 保存助手消息（用户消息已在前面保存）
    add_message(session_id, "assistant", clean_response, role_detected=actual_role)
    st.session_state.messages.append({
        "role": "assistant",
        "content": clean_response,
        "role_detected": actual_role,
    })
