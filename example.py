import asyncio
from os import getcwd
from utcp.utcp_client import UtcpClient
from utcp.data.utcp_client_config import UtcpClientConfigSerializer
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

import json

model = ChatOllama(model="gemma2:9b")
answer_model = ChatOllama(model= "llama3.2:latest")

from langchain_core.prompts import ChatPromptTemplate
from fine_vector import retriever

template = """
    You are an exeprt in answering questions about a vehicle.

    Always find the correct API path from the vehicle API CSV that matches the user's request.  
    - Use the 'prefer_name' column to identify the API path first, as it provides the most human-friendly description.  
    - If multiple APIs match, choose the most relevant one based on the user's request. 
    
    If they ask for api then prioritize giving the whole api and you should you "." in api not "/", however answer as straight to the point as posible.
    
    Don't output anything extra including ``` and the word json.

    Here are some relevant infor: {information}

    Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

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
        user_input = []
        enter = input("Enter: ")
        user_input.append(enter)
        if user_input[-1] == "exit":
            break
        last_message = enter
        information = retriever.invoke(last_message)


        # Assign result to response
        response = chain.invoke({
            "information": information,
            "question": last_message,
        })

        # Create conversation context
        prompt = SystemMessage(
        content="""
                You are my AI assistant.
                You control a vehicle system through VSS APIs.

                Very important rules:
                - The 'api' field must always be in VSS dot notation, like: "Vehicle.Body.Lights.Beam.Low.IsOn".
                - NEVER return URLs like "https://vehicles.com/...".
                - Do not add extra words, comments, or symbols.
                - Respond ONLY with valid JSON in this format:

                    {"body": {"api": "Vehicle.Body.Lights.Beam.Low.IsOn", "value": true}}

                The "value" must be:
                - true when the user wants to turn something ON
                - false when the user wants to turn something OFF
                - or a number when the user wants to set a numeric value.

                Do not include ``` or explanations.
            """
        )


        messages = [f"Detect api based on this information: {response.content}" , prompt, HumanMessage(content=user_input[-1])]
        
        print("response", response.content)
        # Get model response
        result = answer_model.invoke(messages)
        print("AI:", result.content)

        # Check if model requested tool call
        if tools:
            tool_to_call = tools[0].name
            args = json.loads(result.content)

            result = await client.call_tool(tool_to_call, args)
            print(f"\n{tool_to_call}")
            print(result)

        else:
            print("No tools available.")

if __name__ == "__main__":
    asyncio.run(main())
