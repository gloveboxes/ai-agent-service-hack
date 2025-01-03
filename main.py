import os
import asyncio
import logging

from typing import Any, Callable, Dict, Set
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import (
    CodeInterpreterTool,
    FunctionTool,
    AsyncFunctionTool,
    ToolSet,
    AsyncToolSet,
)

from sales_data import SalesData
from my_event_handler import MyEventHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")

AGENT_READY = False

sales_data = SalesData()


user_async_functions: Set[Callable[..., Any]] = {
    sales_data.async_fetch_sales_data_using_sqlite_query,
}


project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)


functions = AsyncFunctionTool(user_async_functions)

definitions = functions.definitions[0]
print(definitions)


code_interpreter = CodeInterpreterTool()

toolset = AsyncToolSet()
toolset.add(functions)
toolset.add(code_interpreter)


async def initialize() -> None:
    """Initialize the assistant with the sales data schema and instructions."""
    global AGENT_READY
    if AGENT_READY:
        return

    await sales_data.connect()
    database_schema_string = await sales_data.get_database_info()

    instructions = (
        "You are an advanced sales analysis assistant for Contoso, specializing in assisting users with sales data inquiries. Maintain a polite, professional, helpful, and friendly demeanor at all times.",

        "Use the `fetch_sales_data_using_sqlite_query` function to execute sales data queries, defaulting to aggregated data unless a detailed breakdown is requested. The function returns JSON-formatted results.",

        f"Refer to the Contoso sales database schema: {database_schema_string}.",

        "When asked for 'help,' provide example queries such as:",
        "- 'What was last quarter's revenue?'",
        "- 'Top-selling products in Europe?'",
        "- 'Total shipping costs by region?'",

        "Responsibilities:",
        "1. Data Analysis: Provide clear insights based on available sales data.",
        "2. Visualizations: Generate charts or graphs to illustrate trends.",
        "3. Scope Awareness:",
        "   - For non-sales-related or out-of-scope questions, reply with:",
        "     'I'm unable to assist with that. Please contact IT for further assistance.'",
        "   - For help requests, suggest actionable and relevant questions.",
        "4. Handling Difficult Interactions:",
        "   - Remain calm and professional when dealing with upset or hostile users.",
        "   - Respond with: 'I'm here to help with your sales data inquiries. If you need further assistance, please contact IT.'",

        "Conduct Guidelines:",
        "- Always maintain a professional and courteous tone.",
        "- Only use data from the Contoso sales database.",
        "- Avoid sharing sensitive or confidential information.",
        "- For questions outside your expertise or unclear queries, respond with:",
        "  'I'm unable to assist with that. Please ask more specific questions about Contoso sales or contact IT for help.'",
    )

    # Simple example of creating an agent and sending a message

    try:
        agent = await project_client.agents.create_agent(
            model=API_DEPLOYMENT_NAME,
            name="Contoso Sales Assistant",
            instructions=str("You are a helpful assistant"),
        )
        print(f"Created agent, ID: {agent.id}")

        thread = await project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        message = await project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content="what is love in 100 words",
        )

        print(f"Created message, ID: {message.id}")

        stream = await project_client.agents.create_stream(
            thread_id=thread.id,
            assistant_id=agent.id,
            event_handler=MyEventHandler(
                functions=functions, project_client=project_client),
            max_completion_tokens=10480,
            max_prompt_tokens=10480,
        )

        async with stream as s:
            await s.until_done()

        await project_client.agents.delete_thread(thread.id)
        await project_client.agents.delete_agent(agent.id)

        # Tool calling example

        # print(str(instructions))

        agent = await project_client.agents.create_agent(
            model=API_DEPLOYMENT_NAME,
            name="Contoso Sales Assistant",
            instructions=str(instructions),
            tools=functions.definitions
        )
        print(f"Created agent, ID: {agent.id}")

        thread = await project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        message = await project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content="What were the total sales by region for 2023",
        )

        print(f"Created message, ID: {message.id}")

        stream = await project_client.agents.create_stream(
            thread_id=thread.id,
            assistant_id=agent.id,
            event_handler=MyEventHandler(
                functions=functions, project_client=project_client),
            max_completion_tokens=4096,
            max_prompt_tokens=4096,
            temperature=0.2,
        )

        async with stream as s:
            await s.until_done()

        await project_client.agents.delete_thread(thread.id)
        await project_client.agents.delete_agent(agent.id)

    except Exception as e:
        logger.error(
            "An error occurred initializing the assistant: %s", str(e))


if __name__ == "__main__":
    print("Starting async program...")
    asyncio.run(initialize())
    print("Program finished.")
    exit(0)
