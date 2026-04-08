from openenv.core.env_server import create_fastapi_app
from models import SupportAction, SupportObservation
from server.environment import SupportEnvironment
from fastapi.responses import HTMLResponse
import uvicorn

dashboard_content = r"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Support Ticket Routing | OpenEnv Dashboard</title>
    <style>
        :root {
            --primary: #2563eb; --primary-dark: #1e40af; --success: #10b981; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; --accent: #6366f1;
        }
        body { font-family: 'Inter', -apple-system, sans-serif; background-color: var(--bg); color: var(--text); line-height: 1.6; margin: 0; padding: 2rem; }
        .container { max-width: 1000px; margin: 0 auto; }
        .card { background: var(--card); padding: 2.5rem; border-radius: 1rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; margin-bottom: 2rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        h1 { color: var(--primary); margin-top: 0; font-size: 2.25rem; letter-spacing: -0.025em; }
        h2 { font-size: 1.5rem; color: var(--primary-dark); border-bottom: 2px solid #f1f5f9; padding-bottom: 0.5rem; margin-top: 1.5rem; }
        .status-badge { display: inline-flex; align-items: center; background: #ecfdf5; color: var(--success); padding: 0.5rem 1rem; border-radius: 9999px; font-weight: 600; font-size: 0.875rem; margin-bottom: 1.5rem; }
        .status-dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; margin-right: 8px; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
        code { background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-family: 'Fira Code', monospace; color: #d63384; font-size: 0.9em; }
        pre { background: #1e293b; color: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; overflow-x: auto; font-size: 0.875rem; }
    </style>
</head>
<body>
    <div class='container'>
        <div class='card'>
            <div class='status-badge'><div class='status-dot'></div>API Status: Active & Online</div>
            <h1>Customer Support Ticket Routing</h1>
            <p style='font-size: 1.2rem; color: #64748b;'>Scaler Meta PyTorch Hackathon Submission</p>
            
            <div class='section'>
                <h2>🎯 Environment Architecture</h2>
                <p>A specialized Reinforcement Learning environment designed to measure LLM agent accuracy in organizational triage. This environment facilitates high-precision routing of complex support tickets.</p>
                <div class='grid'>
                    <div>
                        <strong>Action Space:</strong>
                        <ul>
                            <li><code>route</code>: Direct ticket to a department.</li>
                            <li><code>search</code>: Query internal knowledge base.</li>
                        </ul>
                    </div>
                    <div>
                        <strong>Departments:</strong>
                        <ul>
                            <li><code>Billing</code>: Invoices & Payments</li>
                            <li><code>Tech</code>: Debugging & API Errors</li>
                            <li><code>Sales</code>: Enterprise Pricing</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class='section'>
                <h2>🧪 Technical Specs (For Judges)</h2>
                <p>Integration with this environment is standard via <code>openenv-core</code>. The following endpoint structure is exposed:</p>
                <ul>
                    <li><strong>WebSocket:</strong> <code>/ws</code> (Main interaction loop)</li>
                    <li><strong>State:</strong> <code>/state</code> (Retrieve SupportState)</li>
                    <li><strong>Reset:</strong> <code>/reset</code> (Initialize task)</li>
                </ul>
                <pre># Connect using Python SDK
client = SupportEnvClient(base_url='YOUR_SPACE_URL').sync()
obs = client.reset(task_name='hard')
result = client.step(SupportAction(action_type='route', department='Tech'))</pre>
            </div>

            <div class='section'>
                <h2>🏆 Difficulty & Rewards</h2>
                <p>Tasks are weighted by the complexity of ticket content and required lookups.</p>
                <ul>
                    <li><strong>Easy:</strong> High confidence keywords (1 ticket).</li>
                    <li><strong>Medium:</strong> Conversational support language (2 tickets).</li>
                    <li><strong>Hard:</strong> Raw API logs and technical stack traces (3 tickets).</li>
                </ul>
                <p><strong>Reward Signaling:</strong> This environment utilizes a <code>[0.01, 0.99]</code> bounded reward system to provide a clean signal for RL training while maintaining strict judge-compliant bounds.</p>
            </div>
        </div>
        <div class='footer' style='text-align: center; color: #94a3b8; font-size: 0.875rem;'>
            Powered by OpenEnv SDK
        </div>
    </div>
</body>
</html>
"""

app = create_fastapi_app(SupportEnvironment, SupportAction, SupportObservation)

@app.get('/', response_class=HTMLResponse)
@app.get('/web', response_class=HTMLResponse)
async def root():
    return dashboard_content

def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
