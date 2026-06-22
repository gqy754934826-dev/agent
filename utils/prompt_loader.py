from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_base_prompt():
    try:
        path = get_abs_path(prompts_conf["main_prompt_path"])
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_base_prompt]加载基础提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]加载RAG提示词出错，{str(e)}")
        raise e


def load_expert_prompt(role: str) -> str:
    try:
        expert_prompts = prompts_conf.get("expert_prompts", {})
        if role not in expert_prompts:
            logger.warning(f"[load_expert_prompt]未找到角色{role}的提示词，使用基础提示词")
            return load_base_prompt()

        path = get_abs_path(expert_prompts[role])
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_expert_prompt]加载{role}专家提示词出错，{str(e)}")
        return load_base_prompt()
