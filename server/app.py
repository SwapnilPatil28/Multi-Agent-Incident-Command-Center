from openenv.core.env_server import create_fastapi_app
from models import SupportAction, SupportObservation
from server.environment import SupportEnvironment
import uvicorn

app = create_fastapi_app(SupportEnvironment, SupportAction, SupportObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
