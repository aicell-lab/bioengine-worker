from hypha_rpc import login, connect_to_server
import asyncio

def hello_world(context):
    return "Hello World"
def run_python_script(script, data_url, context):
    pass
async def register_service():
    server_url = "https://hypha.aicell.io"
    workspace_name = "ws-user-google-oauth2|***" # TODO: Change
    service_id = "ray"
    client_id = "berzelius"

    # Login to hypha server
    token = await login({"server_url": server_url})

    # Connect to the workspace
    colab_client = await connect_to_server(
        {
            "server_url": server_url,
            "workspace": workspace_name,
            "client_id": client_id,
            "name": "Berzelius",
            "token": token,
        }
    )

    # Register a new service
    service_info = await colab_client.register_service(
        {
            "name": "ray",
            "id": service_id,
            "config": {
                "visibility": "public",
                "require_context": True,  # TODO: only allow the service to be called by logged-in users
            },
            # Exposed functions:
            "hello_world": hello_world,
            "run_python_script": run_python_script,

        }, {"overwrite": True}
    )
    sid = service_info["id"]
    assert sid == f"{workspace_name}/{client_id}:{service_id}"
    print(f"Registered service with ID: {sid}")
    print(f"Test the service at: {server_url}/{workspace_name}/services/{client_id}:{service_id}/hello_world")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(register_service())
    loop.run_forever()
