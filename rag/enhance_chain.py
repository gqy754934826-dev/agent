from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

from rag.file_history_store import get_history
from rag.vector_store import VectorStoreService
from model.factory import chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.prompt_loader import load_expert_prompt


def format_docs(docs: list[Document]):
    if not docs:
        return "无参考资料"
    formatted_str = ""
    for doc in docs:
        formatted_str += f"文档片段：{doc.page_content}\n文档来源：{doc.metadata}\n\n"
    return formatted_str


# 有参考资料时的提示词（各角色共用）
CONTEXT_PROMPT = "以我提供的已知参考资料为主，简洁和专业的回答用户的问题。参考资料：{context}"


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService()

    def _build_chain(self, with_context: bool, role: str = "base"):
        retriever = self.vector_service.get_retriever()

        # 根据角色加载对应提示词
        if role != "base":
            system_prompt = load_expert_prompt(role)
        else:
            system_prompt = load_expert_prompt("base")

        if with_context:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("system", CONTEXT_PROMPT),
                ("system", "并且我提供用户的对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答我{input}")
            ])
            chain = (
                RunnablePassthrough.assign(context=itemgetter("input") | retriever | format_docs)
                | prompt
                | chat_model
                | StrOutputParser()
            )
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("system", "并且我提供用户的对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user", "{input}")
            ])
            chain = prompt | chat_model | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def stream(self, query: str, context: str, session_id: str, role: str = "base"):
        with_context = bool(context and context != "无参考资料")
        chain = self._build_chain(with_context, role)
        session_config = {"configurable": {"session_id": session_id}}
        data = {"input": query, "context": context} if with_context else {"input": query}
        return chain.stream(data, session_config)

