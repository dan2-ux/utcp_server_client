from fastapi import FastAPI
from pydantic import BaseModel
from utcp_http.http_call_template import HttpCallTemplate
from utcp.data.utcp_manual import UtcpManual
from utcp.python_specific_tooling.tool_decorator import utcp_tool
import uvicorn

from kuksa_client.grpc.aio import VSSClient
from kuksa_client.grpc import Datapoint
from typing import Union
import json

class TestRequest(BaseModel):
    api: str
    value: Union[bool, int, str]

__version__ = "1.0.0"
BASE_PATH = "http://localhost:8080"

try:
    with open('define.json') as F:
        configure = json.load(F)
except Exception as e:
    print("Error: " , e)

vss = VSSClient(configure["ip_address"], configure["port"])

app = FastAPI()

@app.get("/utcp8", response_model=UtcpManual)
def get_utcp():
    return UtcpManual.create_from_decorators(manual_version=__version__) 

@utcp_tool(tool_call_template=HttpCallTemplate(
    name="setter",
    url=f"{BASE_PATH}/setter",
    http_method="POST"
))

@app.post("/setter")
async def target_value_setter(tool : TestRequest):
    try:
        async with vss as client:
            success = await client.set_target_values({
                tool.api: Datapoint(tool.value)
            })
            return success
    except:
        return f"Failed to set {tool.api} value to {tool.value}"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
