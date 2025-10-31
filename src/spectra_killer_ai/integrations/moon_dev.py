"""
Moon Dev AI Agents Integration
Integration layer for Moon Dev swarm intelligence and RBI backtesting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AIModel(Enum):
    """AI models for swarm consensus"""
    CLAUDE = "anthropic/claude-3-5-sonnet"
    GPT = "openai/gpt-4o"
    GEMINI = "google/gemini-2.0-flash"
    GROK = "xai/grok-beta"
    DEEPSEEK = "deepseek/deepseek-chat"
    DEEPSEEK_R1 = "deepseek/deepseek-reasoner"


@dataclass
class AIResponse:
    """Single AI model response"""
    model: AIModel
    signal: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    metadata: Dict


@dataclass
class SwarmConsensus:
    """Swarm consensus result"""
    signal: str
    confidence: float
    model_responses: List[AIResponse]
    consensus_ratio: float
    timestamp: datetime
    

class MoonDevSwarmIntegration:
    """Integration with Moon Dev swarm agent system"""
    
    def __init__(self, config: Dict):
        """
        Initialize Moon Dev swarm integration
        
        Args:
            config: Integration configuration
        """
        self.config = config
        self.enabled_models = config.get('enabled_models', [AIModel.CLAUDE, AIModel.GPT])
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        self.timeout = config.get('timeout', 45)
        
        # API keys (from config or environment)
        self.api_keys = config.get('api_keys', {})
        
        logger.info(f"MoonDevSwarmIntegration initialized with {len(self.enabled_models)} models")
    
    async def get_swarm_consensus(self, market_data: Dict, strategy_context: Dict) -> SwarmConsensus:
        """
        Get swarm consensus from multiple AI models
        
        Args:
            market_data: Current market data
            strategy_context: Strategy context and parameters
            
        Returns:
            SwarmConsensus: Consensus result
        """
        model_responses = []
        
        # Query all enabled models in parallel
        tasks = []
        for model in self.enabled_models:
            task = self._query_model(model, market_data, strategy_context)
            tasks.append(task)
        
        # Wait for all models to respond
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Model {self.enabled_models[i].value} failed: {response}")
                    continue
                if response:
                    model_responses.append(response)
                    
        except Exception as e:
            logger.error(f"Swarm consensus failed: {e}")
        
        # Calculate consensus
        return self._calculate_consensus(model_responses)
    
    async def _query_model(self, model: AIModel, market_data: Dict, strategy_context: Dict) -> Optional[AIResponse]:
        """Query individual AI model for signal"""
        try:
            # Construct prompt based on model type and context
            prompt = self._construct_prompt(model, market_data, strategy_context)
            
            # Simulate API call (replace with actual implementation)
            await asyncio.sleep(0.5)  # Simulate network latency
            
            # Mock response for demonstration
            if "RSI" in str(market_data) and market_data.get("rsi", 50) < 30:
                signal = "BUY"
                confidence = 0.75
                reasoning = f"RSI at {market_data.get('rsi', 50)} indicates oversold condition"
            elif "RSI" in str(market_data) and market_data.get("rsi", 50) > 70:
                signal = "SELL"
                confidence = 0.75
                reasoning = f"RSI at {market_data.get('rsi', 50)} indicates overbought condition"
            else:
                signal = "HOLD"
                confidence = 0.5
                reasoning = "No clear trading signal based on current indicators"
            
            return AIResponse(
                model=model,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
        except Exception as e:
            logger.error(f"Query {model.value} failed: {e}")
            return None
    
    def _construct_prompt(self, model: AIModel, market_data: Dict, strategy_context: Dict) -> str:
        """Construct model-specific prompt"""
        base_prompt = f"""
        Analyze the following trading data for XAUUSD (Gold):
        
        Current Market Data:
        - Price: {market_data.get('price', 'N/A')}
        - RSI: {market_data.get('rsi', 'N/A')}
        - EMA Short: {market_data.get('ema_short', 'N/A')}
        - EMA Long: {market_data.get('ema_long', 'N/A')}
        
        Strategy Context:
        - Timeframe: {strategy_context.get('timeframe', 'M5')}
        - Risk Tolerance: {strategy_context.get('risk_tolerance', 'medium')}
        
        Provide a trading signal (BUY/SELL/HOLD) with confidence (0.0-1.0) and reasoning.
        Respond in JSON format: {{"signal": str, "confidence": float, "reasoning": str}}
        """
        
        # Model-specific adjustments
        if model == AIModel.CLAUDE:
            return base_prompt.replace("trading signal", "XAUUSD trading signal")
        elif model == AIModel.GPT:
            return base_prompt.replace("Gold", "XAUUSD Gold")
        elif model == AIModel.DEEPSEEK:
            return base_prompt + "\nFocus on technical analysis accuracy."
        elif model == AIModel.GROK:
            return base_prompt + "\nConsider recent market movements."
        
        return base_prompt
    
    def _calculate_consensus(self, responses: List[AIResponse]) -> SwarmConsensus:
        """Calculate consensus from model responses"""
        if not responses:
            return SwarmConsensus(
                signal="HOLD",
                confidence=0.0,
                model_responses=[],
                consensus_ratio=0.0,
                timestamp=datetime.now()
            )
        
        # Count signal types
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_confidence = 0
        
        for response in responses:
            signal_counts[response.signal] += 1
            total_confidence += response.confidence
        
        # Determine majority signal
        majority_signal = max(signal_counts, key=signal_counts.get)
        consensus_count = signal_counts[majority_signal]
        
        # Calculate consensus ratio and confidence
        consensus_ratio = consensus_count / len(responses)
        avg_confidence = total_confidence / len(responses)
        
        # Final confidence combines consensus ratio and average confidence
        final_confidence = min(consensus_ratio * avg_confidence * 1.5, 1.0)
        
        return SwarmConsensus(
            signal=majority_signal if consensus_ratio >= self.consensus_threshold else "HOLD",
            confidence=final_confidence,
            model_responses=responses,
            consensus_ratio=consensus_ratio,
            timestamp=datetime.now()
        )


class RBIIntegration:
    """Research-Based Inference backtesting integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backtest_threshold = config.get('save_if_over_return', 1.0)
        self.target_return = config.get('target_return', 50.0)
    
    async def research_strategy(self, strategy_idea: str) -> Dict:
        """
        Research and test trading strategy idea
        
        Args:
            strategy_idea: Strategy description in natural language
            
        Returns:
            Dict: Research results and backtest performance
        """
        try:
            # Simulate strategy research
            research_result = {
                "strategy_idea": strategy_idea,
                "status": "researched",
                "backtest_generated": True,
                "performance": {
                    "return": 15.5,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 8.3,
                    "win_rate": 62.5
                },
                "code_generated": self._generate_strategy_code(strategy_idea),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"RBI research completed for: {strategy_idea[:50]}...")
            return research_result
            
        except Exception as e:
            logger.error(f"RBI research failed: {e}")
            return {"error": str(e)}
    
    def _generate_strategy_code(self, strategy_idea: str) -> str:
        """Generate strategy code from research"""
        return f"""
# Auto-generated strategy based on: {strategy_idea[:50]}...

def rbi_strategy(data, config):
    # RSI-based strategy (example generation)
    rsi = calculate_rsi(data, period=14)
    
    buy_signals = rsi < 30
    sell_signals = rsi > 70
    
    return buy_signals, sell_signals
"""


class MoonDevIntegration:
    """Main integration class for Moon Dev functionality"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.swarm = MoonDevSwarmIntegration(config.get('swarm', {}))
        self.rbi = RBIIntegration(config.get('rbi', {}))
        
        logger.info("MoonDev integration initialized")
    
    async def get_enhanced_signal(self, market_data: Dict, strategy_context: Dict) -> Dict:
        """
        Get enhanced trading signal using Moon Dev integrations
        
        Args:
            market_data: Current market data
            strategy_context: Strategy context
            
        Returns:
            Dict: Enhanced signal with consensus and metadata
        """
        # Get swarm consensus
        consensus = await self.swarm.get_swarm_consensus(market_data, strategy_context)
        
        return {
            "signal": consensus.signal,
            "confidence": consensus.confidence,
            "source": "moon_dev_swarm",
            "consensus_ratio": consensus.consensus_ratio,
            "model_count": len(consensus.model_responses),
            "model_responses": [
                {
                    "model": resp.model.value,
                    "signal": resp.signal,
                    "confidence": resp.confidence,
                    "reasoning": resp.reasoning
                }
                for resp in consensus.model_responses
            ],
            "timestamp": consensus.timestamp.isoformat()
        }
    
    async def research_strategy(self, strategy_idea: str) -> Dict:
        """Research strategy using RBI integration"""
        return await self.rbi.research_strategy(strategy_idea)
