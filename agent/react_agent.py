from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_base_prompt
from agent.tools.agent_tools import rag_search, get_current_time, switch_to_expert
from agent.tools import agent_tools
from rag.enhance_chain import RagService
from agent.tools.middleware import monitor_tool, log_before_model, expert_prompt_switch
from utils.logger_handler import logger
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage, ToolMessageChunk


class XiaoQiAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_base_prompt(),
            tools=[rag_search, get_current_time, switch_to_expert],
            middleware=[monitor_tool, log_before_model, expert_prompt_switch],
        )
        self.rag_service = RagService()

    def execute_stream(self, query: str, session_id: str = "", history: list = None):
        agent_tools.current_session_id = session_id

        messages = []
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append({"role": "user", "content": query})
        input_dict = {"messages": messages}

        rag_context = ""
        actual_role = "base"

        for msg_chunk, metadata in self.agent.stream(
            input_dict,
            stream_mode="messages",
            context={"role": "base"},
        ):
            if isinstance(msg_chunk, (ToolMessage, ToolMessageChunk)):
                content = msg_chunk.content if isinstance(msg_chunk.content, str) else str(msg_chunk.content)
                if "无参考资料" not in content and "已切换到" not in content and content.strip():
                    rag_context = content
                if "已切换到" in content and "专家模式" in content:
                    import re
                    m = re.search(r'已切换到(\w+)专家模式', content)
                    if m:
                        actual_role = m.group(1)
                        yield f"__role__:{actual_role}"
                continue

        # 所有回答都走增强链，传入角色加载对应提示词
        for token in self.rag_service.stream(query, rag_context, session_id, role=actual_role):
            yield token


if __name__ == '__main__':
    agent = XiaoQiAgent()
    for chunk in agent.execute_stream("你好，你是谁？"):
        print(chunk, end="", flush=True)
