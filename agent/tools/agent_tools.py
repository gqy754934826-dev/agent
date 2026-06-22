from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.vector_store import VectorStoreService
from datetime import datetime

vector_service = VectorStoreService()

current_session_id = ""


@tool(description="从向量存储中检索相关参考资料，入参为query（检索词），返回原始参考资料文本")
def rag_search(query: str) -> str:
    logger.info(f"[rag_search]query: {query}")
    retriever = vector_service.get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "无参考资料"
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"【参考资料{i}】：{doc.page_content}\n"
    logger.info(f"[rag_search]检索完成，{len(docs)}条结果")
    return context


@tool(description="获取当前日期和时间，无入参")
def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool(description="切换专家角色提示词，入参为role（角色名），可选值：code/psychology/history/literature/science。")
def switch_to_expert(role: str) -> str:
    valid_roles = ["code", "psychology", "history", "literature", "science"]
    if role not in valid_roles:
        return f"无效的角色：{role}，可选值：{', '.join(valid_roles)}"
    return f"已切换到{role}专家模式"
