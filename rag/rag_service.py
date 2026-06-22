"""
向量检索服务：只负责从向量库检索原始参考资料
"""
from rag.vector_store import VectorStoreService
from langchain_core.documents import Document


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService()
        self.retriever = self.vector_service.get_retriever()

    def retrieve_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def get_context(self, query: str) -> str:
        """检索并格式化为参考资料字符串"""
        docs = self.retrieve_docs(query)
        if not docs:
            return "无参考资料"
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"【参考资料{i}】：{doc.page_content}\n"
        return context
