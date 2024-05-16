#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/5/16 13:59
# @File  : awel_simple_operator.py
# @Author:
# @Desc  : """awel-simple-operator operator package"""
import asyncio
from dbgpt.core.awel import MapOperator, DAG
from dbgpt.core.awel.flow import ViewMetadata, OperatorCategory, IOField, Parameter


class SimpleTwoOperator(MapOperator[str, str]):
    # The metadata for AWEL flow
    metadata = ViewMetadata(
        label="Simple Two Operator",
        name="simple_two_operator",
        category=OperatorCategory.COMMON,
        description="useless description",
        parameters=[
            Parameter.build_from(
                "Name",
                "name",
                str,
                optional=True,
                default="How Are you!",
                description="how are you ",
            )
        ],
        inputs=[
            IOField.build_from(
                "Input value",
                "value",
                str,
                default="Xiao Peng You!",
                description="some people",
            )
        ],
        outputs=[
            IOField.build_from(
                "Output value",
                "value",
                str,
                description="输出这些组合的内容",
            )
        ]
    )

    def __init__(self, name: str = "World", **kwargs):
        super().__init__(**kwargs)
        self.name = name

    async def map(self, value: str) -> str:
        return f"Hello, {self.name}! {value}"

if __name__ == '__main__':
    with DAG("awel_two_operator") as dag:
        task = SimpleTwoOperator()
    result = asyncio.run(task.call(call_data="world"))
    print(result)
