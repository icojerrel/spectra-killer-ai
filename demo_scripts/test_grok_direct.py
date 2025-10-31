#!/usr/bin/env python3
"""
WERKELIJKE GROK INTEGRATIE TEST - Geen MT5 nodig
Dit test de EXACTE API call die je wil maken
"""

import requests
import json
from datetime import datetime

print("="*70)
print("✅ GROK DIRECT API TEST - Always Works™ Testing")
print("="*70)

# Simuleer marktdata (zoals MT5 zou geven)
market_data = {
    "symbol": "XAUUSD",
    "price": 2659.80,
    "bid": 2659.75,
    "ma20": 2645.30,
    "ma50": 2638.50,
    "h24": 2665.50,
    "l24": 2652.00,
    "closes_5": [2658.50, 2659.10, 2659.50, 2659.80, 2659.85],
    "account_balance": 50000.00,
    "timestamp": datetime.now().isoformat()
}

print("\n📊 MARKET DATA:")
print(json.dumps(market_data, indent=2))

# API Details
api_key = "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"
api_url = "https://openrouter.ai/api/v1/chat/completions"

# Try different Grok model IDs
models_to_try = [
    "x-ai/grok-code-fast-1",
    "x-ai/grok-2",
    "x-ai/grok-vision-beta",
]

prompt = f"""You are a professional trading AI analyzing XAUUSD.

Current Market:
- Price: ${market_data['price']:.2f}
- MA20: ${market_data['ma20']:.2f}
- MA50: ${market_data['ma50']:.2f}
- Account: ${market_data['account_balance']:.0f}

Provide trading decision in JSON format:
{{"decision": "BUY" or "SELL" or "HOLD", "confidence": 0.0-1.0, "reasoning": "brief"}}

IMPORTANT: Respond ONLY with valid JSON, no markdown or explanation."""

print("\n" + "="*70)
print("🔴 TESTING GROK MODELS VIA OPENROUTER")
print("="*70)

for model in models_to_try:
    print(f"\n🧪 Testing: {model}")
    print("-" * 70)
    
    try:
        print(f"   📤 Sending request to OpenRouter...")
        
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://spectra-killer-ai.local",
                "X-Title": "Spectra Killer AI"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 200,
                "top_p": 1.0
            },
            timeout=30
        )
        
        print(f"   📬 Status Code: {response.status_code}")
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"   ✅ SUCCESS!")
                print(f"   📝 Response: {content[:250]}...")
                
                # Try to parse JSON
                try:
                    # Extract JSON from response (in case it has markdown)
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        decision = json.loads(json_str)
                        
                        print(f"\n   ✅ PARSED DECISION:")
                        print(f"      Signal: {decision.get('decision', 'N/A')}")
                        print(f"      Confidence: {decision.get('confidence', 0):.1%}")
                        print(f"      Reasoning: {decision.get('reasoning', 'N/A')}")
                        
                        if decision.get('confidence', 0) >= 0.60:
                            print(f"      🎯 ACTION: EXECUTE {decision.get('decision')}")
                        else:
                            print(f"      ⏸️  ACTION: SKIP (Low confidence)")
                    else:
                        print(f"   ⚠️  Response doesn't contain JSON")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON Parse Error: {e}")
                    
            else:
                print(f"   ❌ No choices in response: {result}")
                
        elif response.status_code == 400:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            
            if "not a valid model" in error_msg.lower():
                print(f"   ❌ Model not found: {model}")
            else:
                print(f"   ❌ Bad Request: {error_msg}")
                
        elif response.status_code == 401:
            print(f"   ❌ AUTHENTICATION ERROR - API key invalid")
            print(f"      Message: {response.json()}")
            break  # No point trying other models
            
        elif response.status_code == 429:
            print(f"   ❌ RATE LIMITED - Try again later")
            
        else:
            print(f"   ❌ Error {response.status_code}")
            print(f"      {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print(f"   ⏱️  TIMEOUT - API took too long")
    except requests.exceptions.ConnectionError as e:
        print(f"   🌐 CONNECTION ERROR: {e}")
    except Exception as e:
        print(f"   💥 ERROR: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
