from typing import Optional
from dbgpt.agent import (
    Action,
    ActionOutput,
    AgentResource,
    ResourceType,
)
from dbgpt.agent.util import cmp_string_equal
from dbgpt.vis import Vis
from pydantic import BaseModel, Field

class DiscoveryActionInput(BaseModel):
    summary: str = Field(
        ...,
        description="生成的结果",
    )


class DiscoveryAction(Action[DiscoveryActionInput]):
    def __init__(self):
        super().__init__()

    @property
    def resource_need(self) -> Optional[ResourceType]:
        # The resource type that the current Agent needs to use
        # here we do not need to use resources, just return None
        return None

    @property
    def render_protocol(self) -> Optional[Vis]:
        # The visualization rendering protocol that the current Agent needs to use
        # here we do not need to use visualization rendering, just return None
        return None

    @property
    def out_model_type(self):
        return DiscoveryActionInput

    async def run(
            self,
            ai_message: str,
            resource: Optional[AgentResource] = None,
            rely_action_out: Optional[ActionOutput] = None,
            need_vis_render: bool = True,
            **kwargs,
    ) -> ActionOutput:
        """Perform the action.

        The entry point for actual execution of Action. Action execution will be
        automatically initiated after model inference.
        """
        extra_param = kwargs.get("action_extra_param_key", None)
        try:
            # Parse the input message
            param: DiscoveryActionInput = self._input_convert(ai_message,
                                                            DiscoveryActionInput)
        except Exception:
            return ActionOutput(
                is_exe_success=False,
                content="The requested correctly structured answer could not be found, "
                        f"ai message: {ai_message}",
            )
        NOT_CORRECT_MESSAGE = "生成的代码是错误的"
        # Check if the summary content is not related to user questions
        if param.summary and cmp_string_equal(
                param.summary,
                NOT_CORRECT_MESSAGE,
                ignore_case=True,
                ignore_punctuation=True,
                ignore_whitespace=True,
        ):
            return ActionOutput(
                is_exe_success=False,
                content="the provided text content is not related to user questions at "
                        f"all. ai message: {ai_message}",
            )
        else:
            return ActionOutput(
                is_exe_success=True,
                content=param.summary,
            )
