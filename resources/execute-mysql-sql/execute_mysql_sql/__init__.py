"""Simple calculator tool example."""
import pymysql
from typing import Annotated
from dbgpt.agent.resource import tool

@tool
def execute_sql_query(sql_query: Annotated[str, 'sql语句', True],
                      host: Annotated[str, '数据库地址', True],
                      port: Annotated[str, '数据库端口', True],
                      user: Annotated[str, '用户名', True],
                      password: Annotated[str, '密码', True],
                      database: Annotated[str, '数据库名称', True]
                      ):
    # 给定数据库的信息和sql，执行SQL
    conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database)
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        ret = cursor.fetchall()
        status = True
    except pymysql.MySQLError as e:
        print("Error:", e)
        ret = f"执行sql异常，报错{e}"
        status = False
    finally:
        cursor.close()
        conn.close()
    return ret
