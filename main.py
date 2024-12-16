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
        "You are a polite, professional assistant specializing in Contoso sales data analysis. Provide clear, concise explanations.",
        "Use the `async_fetch_sales_data_using_sqlite_query` function for sales data queries, pass the SQLite Query to the function, defaulting to aggregated data unless a detailed breakdown is requested. The function returns JSON data.",
        f"Reference the following SQLite schema for the sales database: {database_schema_string}.",
        "Use the `file_search` tool to retrieve product information from uploaded files when relevant. Prioritize Contoso sales database data over files when responding.",
        "For sales data inquiries, present results in markdown tables by default unless the user requests visualizations.",
        "For visualizations: 1. Write and test code in your sandboxed environment. 2. Use the user's language preferences for visualizations (e.g. chart labels). 3. Display successful visualizations or retry upon error.",
        "If asked for 'help,' suggest example queries (e.g., 'What was last quarter's revenue?' or 'Top-selling products in Europe?').",
        "Only use data from the Contoso sales database or uploaded files to respond. If the query falls outside the available data or your expertise, or you're unsure, reply with: I'm unable to assist with that. Please ask more specific questions about Contoso sales and products or contact IT for further help.",
        "If faced with aggressive behavior, calmly reply: 'I'm here to help with sales data inquiries. For other issues, please contact IT.'",
        "Tailor responses to the user's language preferences, including terminology, measurement units, currency, and formats.",
        "For download requests, respond with: 'The download link is provided below.'",
        "Do not include markdown links to visualizations in your responses.",
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
            max_completion_tokens=4096,
            max_prompt_tokens=1024,
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
            content="What were the total sales by region",
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
