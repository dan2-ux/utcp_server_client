import asyncio
from os import getcwd
from utcp.utcp_client import UtcpClient
from utcp.data.utcp_client_config import UtcpClientConfigSerializer
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

import json

model = ChatOllama(model="gemma3:12b")
answer_model = ChatOllama(model= "gemma2:9b")

from langchain_core.prompts import ChatPromptTemplate
from fine_vector import retriever

import csv

api_data = []
with open("data.csv", newline="") as F:
    reader = csv.DictReader(F)
    for row in reader:
        api_data.append(row)

def find_type_using_api(api_path):
    for entry in api_data:
        if entry["path"] == api_path:
            return entry["type"]

import re
def clean_json_response(s):
    # Remove code fences like ``` or ```json
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

#template = """
#    You are an expert in vehicle APIs.

#    Always find the **exact API path** and its **type** ('actuator', 'sensor', or 'attribute') from the vehicle API CSV that matches the user's request.
#    - Use the 'prefer_name' column to identify the API path first.
#    - Use the 'type' column from the CSV to fill in the API type.
#    - If multiple APIs match, choose the most relevant one.

#    Respond ONLY in the following JSON format, without extra words or explanations:

#    {{
#        "api": "<full API path in dot notation>",
#        "type": "<API type>"
#    }}

#    Here are the relevant APIs: {information}

#    Here is the question: {question}
#"""

template = """
    You are an expert in vehicle APIs.

    Always find the **exact API path** from the vehicle API CSV that matches the user's request.
    - Use the 'prefer_name' column to identify the API path first.
    - If multiple APIs match, choose the most relevant one.

    Respond ONLY in the following JSON format, without extra words or explanations:

    {{
        "api": "<full API path in dot notation>",
    }}

    Here are the relevant APIs: {information}

    Here is the question: {question}
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
        system_prompt = SystemMessage(
            content="""
                You are my AI assistant controlling vehicle VSS APIs.

                Rules:
                1. Use the API exactly as provided.
                2. Use the "type" exactly as provided by the retrieved information — do NOT change it. The "type" is between 3 str "actuator", "sensor" and "attribute". Do not choose anything other then 3 of those.
                3. Respond ONLY in the following JSON format:

                [
                    {"body": {"api": "<api>", "type": "<type>", "value": <true/false/number>}},
                    {"tool": "<setter/teller>"}
                ]

                - Determine "value" based on the user request:
                * true to turn ON
                * false to turn OFF
                * a number if user wants a numeric value

                - Determine "tool":
                * "setter" if user wants to change a value
                * "teller" if user wants to read a value (value should be false)

                Do NOT guess or change the "type". Always use the type from the retrieved API information.
                Do NOT add extra text, explanations, or markdown.
            """
        )

        response = clean_json_response(response.content)

        print("response", response)
        response_json = json.loads(response)  # Convert string to dict
        api_path = response_json["api"]
        
        print(find_type_using_api(api_path))
        messages = [
            SystemMessage(content=f"Detected API: {response}, type from CSV: {find_type_using_api(api_path)}. Use this type exactly in your response JSON."),
            system_prompt,
            HumanMessage(content=user_input[-1])
        ]
        # Get model response

        result = answer_model.invoke(messages)
        result = clean_json_response(result.content)
        result_json = json.loads(result)
        result_json[0]["body"]["type"] = find_type_using_api(api_path)
        print("AI:", result_json)

        # Check if model requested tool call
        if tools:
            if result_json[1]["tool"] == "setter":
                tool_to_call = tools[0].name
                args = result_json[0]
            elif result_json[1]["tool"] == "teller":
                tool_to_call = tools[1].name
                args = result_json[0]

            result = await client.call_tool(tool_to_call, args)
            print(f"\n{tool_to_call}")
            print(result)

        else:
            print("No tools available.")

if __name__ == "__main__":
    asyncio.run(main())
