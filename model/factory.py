from utils.config_handler import rag_conf
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["base_url"],
            streaming=True,
        )


class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        return OpenAIEmbeddings(
            model=rag_conf["openai_model"],
            base_url=rag_conf["openai_api_base"],
            api_key=rag_conf["openai_api_key"],
            chunk_size=1,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingModelFactory().generator()
