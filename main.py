import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx


async def main() -> None:
    # Configure MCP servers (Context7 + Met Museum)
    client = MultiServerMCPClient(
        {
            "context7": {
                "command": "npx",
                "args": [
                    "-y",
                    "-p", "ajv@8",
                    "-p", "ajv-formats@2",
                    "-p", "@upstash/context7-mcp",
                    "context7-mcp",
                ],
                "transport": "stdio",
            },
            "met-museum": {
                "command": "npx",
                "args": ["-y", "metmuseum-mcp"],
                "transport": "stdio",
            },
        }
    )

    # Model (use OpenAI; swap to watsonx_model if rate-limited)
    openai_model = ChatOpenAI(model="gpt-5-nano")

    watsonx_model = ChatWatsonx(
        model_id="ibm/granite-3-3-8b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
    )

    # Get tools from MCP servers
    tools = await client.get_tools()

    # Explicit tool demo (so you can prove tools exist)
    print("Available tools:")
    for t in tools:
        print("-", t.name)

    # Memory + thread id (chat history persists within this run)
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "conversation_id"}}

    # Build agent
    agent = create_react_agent(
        model=openai_model,   # or watsonx_model
        tools=tools,
        checkpointer=checkpointer,
    )

    # Initial message: system role + user intro request
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a smart, useful agent with tools to access code library "
                        "documentation and the Met Museum collection."
                    ),
                },
                {
                    "role": "user",
                    "content": "Give a brief introduction of what you do and the tools you can access.",
                },
            ]
        },
        config=config,
    )

    print("\nAgent introduction:")
    print(response["messages"][-1].content)

    	# Main interaction loop - allows continuous conversation with the agent
    while True:
        # Display menu options to the user
        choice = input("""
    Menu:
    1. Ask the agent a question
    2. Quit
    Enter your choice (1 or 2): """)

        if choice == "1":
            # Get user's question
            print("Your question")
            query = input("> ")

            # Send the user's question to the agent
            # The agent will have access to the full conversation history
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},        # User's current question
                config=config              # Maintains conversation thread
            )
            # Display the agent's response
            print(response['messages'][-1].content)
        else:
            # Exit the program for any choice other than "1"
            print("Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(main())