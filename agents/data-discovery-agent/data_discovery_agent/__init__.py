"""
数据探索Agent
"""

import asyncio
from typing import Any, Dict, Optional, Tuple

from dbgpt.agent import AgentMessage, ConversableAgent, ProfileConfig
from dbgpt.core import ModelMessageRoleType

from .action import DiscoveryAction, NOT_RELATED_MESSAGE

CHECK_RESULT_SYSTEM_MESSAGE = (
    "你是1个代码专家，能够检查Bot根据问题生成的代码是否正确。"
    "你的答案应该遵从下面原则:\n"
    "    Rule 1: 如果你认为代码没问任何问题，只需返回True"
    "    Rule 2: 如果你认为生成的代码存在问题，请返回False和问题描述，描述应该包括问题本身和代码中可能存在的问题，并以|分割，最后以TERMINATE结尾"
)


class DataDiscoverAgent(ConversableAgent):
    profile: ProfileConfig = ProfileConfig(
        # The name of the agent
        name="disagent",
        # The role of the agent, 对应DB-GPT:dbgpt/agent/core/agent_manage.py中的def get_by_name(self, name: str)的name
        role="Discovers",
        # The core functional goals of the agent tell LLM what it can do with it.
        goal=(
            "根据提供的数据库，文本等信息，对这些资源进行探索，根据已经提供的工具，写出python Tools和Solution的代码，实现对资源的探索。"
        ),
        # Introduction and description of the agent, used for task assignment and
        # display. If it is empty, the goal content will be used.
        desc=(
            "可以对提供的数据库，文本等信息进行探索。"
        ),
        constraints=[
            """你可以自己创建工具或者使用已有的工具解决问题。
            **Information**
            {{information}}
            **HasTools**
            {{hastools}
            **Question**
            """,
        ],
        examples = ("""
            ## Example
            **Information**
            mysql数据库，ip: 192.168.50.189, 端口:3306, 用户名root, 密码test, 数据库名称taobao
            
            **HasTools**
            def execute_sql_query(sql_query: Annotated[str, 'sql语句', True],
                                  host: Annotated[str, '数据库地址', True],
                                  port: Annotated[str, '数据库端口', True],
                                  user: Annotated[str, '用户名', True],
                                  password: Annotated[str, '密码', True],
                                  database: Annotated[str, '数据库名称', True]
                                  ):
            **Question**
            探索给定的数据库的表结构信息
            
            **CreateTools**
            ```python
            import execute_sql_query
            def show_tables(host,port,user,password,database):
                # SQL query to retrieve table information
                sql_query = "SHOW TABLES"
            
                # Execute SQL query using the function
                tables = execute_sql_query(sql_query,host=host, port=port, user=user, password=password, database=database)
            
                if tables:
                    for table in tables:
                        table_name = table[0]
                        print("Table:", table_name)
                        
                        # SQL query to describe table structure
                        describe_query = "DESCRIBE " + table_name
                        
                        # Execute SQL query using the function
                        table_structure = execute_sql_query(describe_query)
                        
                        # Print table structure
                        if table_structure:
                            for field in table_structure:
                                print(field)
                                
                        print("\n")
            ```
   
            **Solution**
            ```python
            host = '192.168.50.189'
            port = 3306
            user = 'root'
            password = 'test'
            database = 'taobao'
            show_tao_tables(host,port,user,password,database)
            ```
        """)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([DiscoveryAction])

    def _init_reply_message(self, received_message: AgentMessage) -> AgentMessage:
        reply_message = super()._init_reply_message(received_message)
        print(f"收到的received_message: {received_message}")
        # Fill in the dynamic parameters in the prompt template≈
        # information是用户提供的上下信息
        # hastools根据生成的问题检索需要使用的工具
        reply_message.context = {"information": "hello", "hastools": "tools"}
        return reply_message

    def prepare_act_param(self) -> Dict[str, Any]:
        return {"action_extra_param_key": "this is extra param"}

    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        current_goal = message.current_goal
        action_report = message.action_report
        task_result = ""
        if action_report:
            task_result = action_report.get("content", "")

        check_result, model = await self.thinking(
            messages=[
                AgentMessage(
                    role=ModelMessageRoleType.HUMAN,
                    content=(
                        "请检查用户的问题和生成的代码，检查生成的代码是否正确\n"
                        f"用户输入: {current_goal}\n"
                        f"生成结果: {task_result}"
                    ),
                )
            ],
            prompt=CHECK_RESULT_SYSTEM_MESSAGE,
        )

        fail_reason = ""
        if check_result and (
            "true" in check_result.lower() or "yes" in check_result.lower()
        ):
            success = True
        else:
            success = False
            try:
                _, fail_reason = check_result.split("|")
                fail_reason = (
                    "The summary results cannot summarize the user input due"
                    f" to: {fail_reason}. Please re-understand and complete the summary"
                    " task."
                )
            except Exception:
                fail_reason = (
                    "The summary results cannot summarize the user input. "
                    "Please re-understand and complete the summary task."
                )
        return success, fail_reason


async def _test_agent():
    """Test the summarizer agent."""
    from dbgpt.model.proxy import OpenAILLMClient
    from dbgpt.agent import AgentContext, AgentMemory, UserProxyAgent, LLMConfig

    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    context: AgentContext = AgentContext(conv_id="summarize")

    agent_memory: AgentMemory = AgentMemory()

    summarizer = (
        await DataDiscoverAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .build()
    )

    user_proxy = await UserProxyAgent().bind(agent_memory).bind(context).build()

    await user_proxy.initiate_chat(
        recipient=summarizer,
        reviewer=user_proxy,
        message="""I want to summarize advantages of Nuclear Power according to the following content.
            Nuclear power in space is the use of nuclear power in outer space, typically either small fission systems or radioactive decay for electricity or heat. Another use is for scientific observation, as in a Mössbauer spectrometer. The most common type is a radioisotope thermoelectric generator, which has been used on many space probes and on crewed lunar missions. Small fission reactors for Earth observation satellites, such as the TOPAZ nuclear reactor, have also been flown.[1] A radioisotope heater unit is powered by radioactive decay and can keep components from becoming too cold to function, potentially over a span of decades.[2]
            The United States tested the SNAP-10A nuclear reactor in space for 43 days in 1965,[3] with the next test of a nuclear reactor power system intended for space use occurring on 13 September 2012 with the Demonstration Using Flattop Fission (DUFF) test of the Kilopower reactor.[4]
            After a ground-based test of the experimental 1965 Romashka reactor, which used uranium and direct thermoelectric conversion to electricity,[5] the USSR sent about 40 nuclear-electric satellites into space, mostly powered by the BES-5 reactor. The more powerful TOPAZ-II reactor produced 10 kilowatts of electricity.[3]
            Examples of concepts that use nuclear power for space propulsion systems include the nuclear electric rocket (nuclear powered ion thruster(s)), the radioisotope rocket, and radioisotope electric propulsion (REP).[6] One of the more explored concepts is the nuclear thermal rocket, which was ground tested in the NERVA program. Nuclear pulse propulsion was the subject of Project Orion.[7]
            Regulation and hazard prevention[edit]
            After the ban of nuclear weapons in space by the Outer Space Treaty in 1967, nuclear power has been discussed at least since 1972 as a sensitive issue by states.[8] Particularly its potential hazards to Earth's environment and thus also humans has prompted states to adopt in the U.N. General Assembly the Principles Relevant to the Use of Nuclear Power Sources in Outer Space (1992), particularly introducing safety principles for launches and to manage their traffic.[8]
            Benefits
            Both the Viking 1 and Viking 2 landers used RTGs for power on the surface of Mars. (Viking launch vehicle pictured)
            While solar power is much more commonly used, nuclear power can offer advantages in some areas. Solar cells, although efficient, can only supply energy to spacecraft in orbits where the solar flux is sufficiently high, such as low Earth orbit and interplanetary destinations close enough to the Sun. Unlike solar cells, nuclear power systems function independently of sunlight, which is necessary for deep space exploration. Nuclear-based systems can have less mass than solar cells of equivalent power, allowing more compact spacecraft that are easier to orient and direct in space. In the case of crewed spaceflight, nuclear power concepts that can power both life support and propulsion systems may reduce both cost and flight time.[9]
            Selected applications and/or technologies for space include:
            Radioisotope thermoelectric generator
            Radioisotope heater unit
            Radioisotope piezoelectric generator
            Radioisotope rocket
            Nuclear thermal rocket
            Nuclear pulse propulsion
            Nuclear electric rocket
            """,
    )
    print(await agent_memory.gpts_memory.one_chat_completions("summarize"))


if __name__ == "__main__":
    asyncio.run(_test_agent())
