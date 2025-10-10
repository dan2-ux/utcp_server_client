import asyncio
from os import getcwd
from utcp.utcp_client import UtcpClient
from utcp.data.utcp_client_config import UtcpClientConfigSerializer
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

import json

model = ChatOllama(model="gemma2:9b")
answer_model = ChatOllama(model= "llama3.2:latest")

async def main():
    client: UtcpClient = await UtcpClient.create(
        root_dir=getcwd(),
        config=UtcpClientConfigSerializer().validate_dict(
            {
                "manual_call_templates": [
                    {
                        "name": "api",
                        "call_template_type": "http",
                        "http_method": "GET",
                        "url": "http://localhost:8080/utcp8"
                    }
                ]
            }
        )
    )

    # List all available tools
    print("Registered tools:")
    tools = await client.search_tools("")
    for tool in tools:
        print(f" - {tool.name}")

    while True:
        user_input = input("Enter: ")

        if user_input == "exit":
            break

        # Create conversation context
        prompt = SystemMessage(
            content="""
                You are my AI assistant.
                If the user wants change or to alter vehicle api, then spend time to find the api according to the user command then when finish finding api, respond ONLY with JSON like:
                    {"body": {"api": "Vehicle.Body.Lights.Beam.Low.IsOn", "value": true  # true for on, false for off}}
                Do not include any other text, comment or charater. Just straight up json.
            """
        )

        messages = [prompt, HumanMessage(content=user_input)]

        # Get model response
        response = model.invoke(messages)
        print("AI:", response.content)

        # Check if model requested tool call
        if tools:
            tool_to_call = tools[0].name
            args = json.loads(response.content)

            result = await client.call_tool(tool_to_call, args)
            print(f"\n{tool_to_call}")
            print(result)

        else:
            print("No tools available.")

if __name__ == "__main__":
    asyncio.run(main())
