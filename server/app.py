from openenv.core.env_server import create_fastapi_app
from models import IncidentAction, IncidentObservation
from server.environment import IncidentCommandCenterEnvironment
from fastapi.responses import HTMLResponse
import uvicorn

dashboard_content = r"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Incident Command Center | OpenEnv Dashboard</title>
    <style>
        :root { --primary: #3b82f6; --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; }
        body { font-family: -apple-system, sans-serif; background-color: var(--bg); color: var(--text); padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; background: var(--card); padding: 2rem; border-radius: 1rem; }
        code { background: #334155; padding: 0.2rem; border-radius: 0.25rem; font-family: monospace; color: #38bdf8; }
    </style>
</head>
<body>
    <div class='container'>
        <h1>Multi-Agent Incident Command Center</h1>
        <p>Round-2 themes: Multi-Agent Interactions + World Modeling (Professional Tasks).</p>
        
        <h2>Action Space</h2>
        <ul>
            <li><code>inspect_logs(target)</code></li>
            <li><code>inspect_metrics(target)</code></li>
            <li><code>consult_kb(target)</code></li>
            <li><code>negotiate_handoff(target)</code></li>
            <li><code>apply_fix(resolution_summary)</code></li>
            <li><code>close_incident(root_cause)</code></li>
        </ul>
        
        <h2>Reward Logic</h2>
        <p>Dense reward shaping for clue discovery, team coordination, and efficient resolution under budget + SLA constraints. Correct closure with mitigation gets the highest reward.</p>
    </div>
</body>
</html>
"""

app = create_fastapi_app(
    IncidentCommandCenterEnvironment,
    IncidentAction,
    IncidentObservation,
)

@app.get('/', response_class=HTMLResponse)
@app.get('/web', response_class=HTMLResponse)
async def root():
    return dashboard_content

def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
