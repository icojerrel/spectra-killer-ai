"""
Web dashboard application
Real-time trading monitoring with FastAPI and WebSocket
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..cli import SpectraTradingEngine


class DashboardManager:
    """Manages dashboard connections and data broadcasting"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.engine: SpectraTradingEngine = None
        
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_portfolio_update(self):
        """Send portfolio update to all clients"""
        if self.engine:
            status = self.engine.get_engine_status()
            await self.broadcast({
                'type': 'portfolio_update',
                'timestamp': datetime.now().isoformat(),
                'data': status
            })
    
    async def send_signal_update(self, signal: Dict[str, Any]):
        """Send trading signal to all clients"""
        await self.broadcast({
            'type': 'signal_update',
            'timestamp': datetime.now().isoformat(),
            'data': signal
        })


# Global dashboard manager
dashboard_manager = DashboardManager()


def create_dashboard_app(engine: SpectraTradingEngine = None) -> FastAPI:
    """Create FastAPI dashboard application"""
    
    app = FastAPI(
        title="Spectra Killer AI Dashboard",
        description="Real-time trading monitoring dashboard",
        version="2.0.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set engine reference
    dashboard_manager.engine = engine
    
    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve dashboard HTML"""
        return await _get_dashboard_html()
    
    @app.get("/api/status")
    async def get_status():
        """Get current trading status"""
        if dashboard_manager.engine:
            return dashboard_manager.engine.get_engine_status()
        return {"error": "No trading engine running"}
    
    @app.get("/api/performance")
    async def get_performance():
        """Get performance metrics"""
        if dashboard_manager.engine:
            portfolio_stats = dashboard_manager.engine.portfolio.get_performance_metrics()
            risk_report = dashboard_manager.engine.risk_manager.get_risk_report()
            return {
                'portfolio': portfolio_stats,
                'risk': risk_report,
                'timestamp': datetime.now().isoformat()
            }
        return {"error": "No trading engine running"}
    
    @app.get("/api/positions")
    async def get_positions():
        """Get current positions"""
        if dashboard_manager.engine:
            positions = dashboard_manager.engine.portfolio.get_open_positions()
            return {
                'positions': [pos.to_dict() for pos in positions],
                'count': len(positions),
                'timestamp': datetime.now().isoformat()
            }
        return {"error": "No trading engine running"}
    
    @app.get("/api/history")
    async def get_history():
        """Get trading history"""
        if dashboard_manager.engine:
            history = dashboard_manager.engine.portfolio.equity_curve
            return {
                'equity_curve': [
                    {
                        'timestamp': s.timestamp.isoformat(),
                        'balance': float(s.balance),
                        'equity': float(s.equity),
                        'unrealized_pnl': float(s.unrealized_pnl)
                    } for s in history
                ],
                'count': len(history)
            }
        return {"error": "No trading engine running"}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await dashboard_manager.connect(websocket)
        
        try:
            # Start sending updates
            while True:
                await dashboard_manager.send_portfolio_update()
                await asyncio.sleep(1)  # Send update every second
                
        except WebSocketDisconnect:
            dashboard_manager.disconnect(websocket)
    
    async def _get_dashboard_html() -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spectra Killer AI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @keyframes pulse-green {
            0%, 100% { background-color: rgb(34, 197, 94); }
            50% { background-color: rgb(74, 222, 128); }
        }
        @keyframes pulse-red {
            0%, 100% { background-color: rgb(239, 68, 68); }
            50% { background-color: rgb(248, 113, 113); }
        }
        .status-running { animation: pulse-green 2s infinite; }
        .status-stopped { animation: pulse-red 2s infinite; }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-center mb-2">Spectra Killer AI Dashboard</h1>
            <div class="text-center text-gray-400">Real-time Trading Monitor v2.0</div>
        </header>

        <!-- Status Bar -->
        <div class="bg-gray-800 rounded-lg p-4 mb-6">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                <div>
                    <div class="text-sm text-gray-400">Status</div>
                    <div id="status" class="text-lg font-bold">Loading...</div>
                </div>
                <div>
                    <div class="text-sm text-gray-400">Mode</div>
                    <div id="mode" class="text-lg font-bold">-</div>
                </div>
                <div>
                    <div class="text-sm text-gray-400">Balance</div>
                    <div id="balance" class="text-lg font-bold">$0.00</div>
                </div>
                <div>
                    <div class="text-sm text-gray-400">Daily P&L</div>
                    <div id="daily_pnl" class="text-lg font-bold">$0.00</div>
                </div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <!-- Portfolio Chart -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4">Equity Curve</h2>
                <canvas id="equityChart" width="400" height="200"></canvas>
            </div>

            <!-- Risk Metrics -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-xl font-bold mb-4">Risk Metrics</h2>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Risk Score:</span>
                        <span id="risk_score" class="font-bold">0.0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Risk Level:</span>
                        <span id="risk_level" class="font-bold">Low</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Wind Rate:</span>
                        <span id="win_rate" class="font-bold">0%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Max Drawdown:</span>
                        <span id="max_drawdown" class="font-bold">0%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Open Positions:</span>
                        <span id="positions" class="font-bold">0</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Signal Monitor -->
        <div class="bg-gray-800 rounded-lg p-6 mb-6">
            <h2 class="text-xl font-bold mb-4">Latest Signals</h2>
            <div id="signals" class="space-y-2">
                <div class="text-gray-400">Waiting for signals...</div>
            </div>
        </div>

        <!-- Positions Table -->
        <div class="bg-gray-800 rounded-lg p-6">
            <h2 class="text-xl font-bold mb-4">Current Positions</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-gray-700">
                            <th class="text-left py-2">Symbol</th>
                            <th class="text-left py-2">Type</th>
                            <th class="text-right py-2">Size</th>
                            <th class="text-right py-2">Entry</th>
                            <th class="text-right py-2">Current</th>
                            <th class="text-right py-2">P&L</th>
                            <th class="text-left py-2">Status</th>
                        </tr>
                    </thead>
                    <tbody id="positions_table">
                        <tr>
                            <td colspan="7" class="text-center py-4 text-gray-400">No open positions</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        // Chart setup
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: false,
                        ticks: {
                            color: '#9CA3AF',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        },
                        grid: {
                            color: '#374151'
                        }
                    }
                }
            }
        });

        // WebSocket message handler
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'portfolio_update') {
                updateDashboard(data.data);
            } else if (data.type === 'signal_update') {
                addSignal(data.data);
            }
        };

        function updateDashboard(data) {
            // Update status
            const statusEl = document.getElementById('status');
            const modeEl = document.getElementById('mode');
            const balanceEl = document.getElementById('balance');
            const pnlEl = document.getElementById('daily_pnl');
            
            if (data.running) {
                statusEl.textContent = 'RUNNING';
                statusEl.className = 'text-lg font-bold text-green-400';
            } else {
                statusEl.textContent = 'STOPPED';
                statusEl.className = 'text-lg font-bold text-red-400';
            }
            
            modeEl.textContent = data.mode || '-';
            balanceEl.textContent = '$' + (data.portfolio?.balance || 0).toFixed(2);
            
            const dailyPnl = data.portfolio?.daily_pnl || 0;
            pnlEl.textContent = '$' + dailyPnl.toFixed(2);
            pnlEl.className = dailyPnl >= 0 ? 'text-lg font-bold text-green-400' : 'text-lg font-bold text-red-400';
            
            // Update risk metrics
            if (data.risk) {
                document.getElementById('risk_score').textContent = (data.risk.risk_score || 0).toFixed(1);
                document.getElementById('risk_level').textContent = (data.risk.risk_level || 'Low').toUpperCase();
                document.getElementById('positions').textContent = data.portfolio?.open_positions || 0;
            }
            
            // Update portfolio metrics
            if (data.portfolio) {
                document.getElementById('win_rate').textContent = (data.portfolio.win_rate || 0) + '%';
                document.getElementById('max_drawdown').textContent = (data.portfolio.max_drawdown || 0) + '%';
            }
            
            // Update positions table
            updatePositionsTable(data.portfolio?.positions || []);
            
            // Update chart
            updateChart(data.portfolio?.equity_curve || []);
        }

        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions_table');
            
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-gray-400">No open positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(pos => `
                <tr class="border-b border-gray-700">
                    <td class="py-2">${pos.symbol}</td>
                    <td class="py-2">${pos.position_type}</td>
                    <td class="py-2 text-right">${pos.quantity.toFixed(2)}</td>
                    <td class="py-2 text-right">$${pos.entry_price.toFixed(2)}</td>
                    <td class="py-2 text-right">$${(pos.current_price || pos.entry_price).toFixed(2)}</td>
                    <td class="py-2 text-right ${pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                        $${(pos.unrealized_pnl || 0).toFixed(2)}
                    </td>
                    <td class="py-2">
                        <span class="px-2 py-1 text-xs rounded ${pos.status === 'OPEN' ? 'bg-green-600' : 'bg-red-600'}">
                            ${pos.status}
                        </span>
                    </td>
                </tr>
            `).join('');
        }

        function updateChart(equityCurve) {
            if (equityCurve.length === 0) return;
            
            const labels = equityCurve.map((_, index) => index);
            const data = equityCurve.map(point => point.equity);
            
            // Keep only last 100 points
            const maxPoints = 100;
            const startIndex = Math.max(0, labels.length - maxPoints);
            
            equityChart.data.labels = labels.slice(startIndex);
            equityChart.data.datasets[0].data = data.slice(startIndex);
            equityChart.update('none');
        }

        function addSignal(signal) {
            const signalsDiv = document.getElementById('signals');
            const signalEl = document.createElement('div');
            
            const colorClass = signal.signal === 'BUY' ? 'text-green-400' : 
                             signal.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400';
            
            signalEl.className = 'p-3 bg-gray-700 rounded border-l-4 ' + 
                         (signal.signal === 'BUY' ? 'border-green-400' : 
                          signal.signal === 'SELL' ? 'border-red-400' : 'border-yellow-400');
            
            signalEl.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <span class="font-bold ${colorClass}">${signal.signal}</span>
                        <span class="ml-2 text-gray-400">${signal.symbol || 'XAUUSD'}</span>
                    </div>
                    <div class="text-right text-sm text-gray-400">
                        <div>Confidence: ${(signal.confidence * 100).toFixed(1)}%</div>
                        <div>${new Date().toLocaleTimeString()}</div>
                    </div>
                </div>
            `;
            
            // Insert at the beginning and keep only last 5 signals
            signalsDiv.insertBefore(signalEl, signalsDiv.firstChild);
            while (signalsDiv.children.length > 5) {
                signalsDiv.removeChild(signalsDiv.lastChild);
            }
        }

        // Load initial data
        async function loadInitialData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateDashboard(data);
                
                const historyResponse = await fetch('/api/history');
                const historyData = await historyResponse.json();
                updateChart(historyData.equity_curve || []);
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }

        // Initialize dashboard
        loadInitialData();
    </script>
</body>
</html>
        """


def run_dashboard(engine: SpectraTradingEngine = None, host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard server"""
    app = create_dashboard_app(engine)
    
    print(f"🚀 Starting Spectra Killer AI Dashboard")
    print(f"📊 Dashboard URL: http://localhost:{port}")
    print(f"🔗 WebSocket: ws://localhost:{port}/ws")
    print(f"💡 Press Ctrl+C to stop")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
