# AI Chatbot Guardrails System

A comprehensive safety framework for AI chatbots that implements multiple layers of protection against harmful, inappropriate, or problematic content.

## 🛡️ Overview

The AI Chatbot Guardrails System provides a multi-layered defense mechanism to ensure responsible AI behavior. It operates at both input and output levels, implementing real-time safety checks, rate limiting, and session management.

## 🔧 Key Features

### 1. **Content Filtering**
- **Harmful Content Detection**: Blocks requests for illegal, dangerous, or harmful information
- **Sensitive Topic Handling**: Identifies medical, legal, financial advice requests and adds appropriate disclaimers
- **Pattern-Based Filtering**: Uses regex patterns to detect harmful content categories

### 2. **PII Detection**
- **Personal Information Protection**: Detects SSNs, phone numbers, emails, credit cards
- **Data Privacy**: Prevents processing of personally identifiable information
- **Configurable Patterns**: Customizable regex patterns for different PII types

### 3. **Toxicity Detection**
- **AI-Powered Analysis**: Uses OpenAI's Moderation API for sophisticated toxicity detection
- **Multiple Categories**: Detects hate speech, harassment, violence, self-harm content
- **Confidence Scoring**: Provides confidence levels for safety decisions

### 4. **Rate Limiting**
- **User Session Tracking**: Monitors message frequency per user
- **Configurable Limits**: Customizable messages per minute/hour limits
- **Warning System**: Accumulates warnings before blocking users
- **Session Timeout**: Automatic session resets after inactivity

### 5. **Context Validation**
- **Injection Attack Prevention**: Detects prompt injection and context switching attempts
- **Conversation Length Management**: Prevents context window overflow
- **Context Integrity**: Maintains conversation coherence

### 6. **Output Validation**
- **System Leak Prevention**: Blocks responses containing system prompts or instructions
- **Relevance Checking**: Ensures responses are relevant to user queries
- **Response Length Control**: Manages output length to prevent overwhelming responses

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install openai python-dotenv fastapi uvicorn

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Basic Usage

```python
import asyncio
from openai import AsyncOpenAI
from guardrails_system import GuardrailsSystem

async def main():
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key="your_api_key")
    
    # Create guardrails system
    guardrails = GuardrailsSystem(client)
    
    # Test input safety
    user_id = "user123"
    message = "What is machine learning?"
    conversation_history = []
    
    input_safe, results = await guardrails.check_input(
        user_id=user_id,
        message=message,
        conversation_history=conversation_history
    )
    
    if input_safe:
        print("✅ Message is safe to process")
        
        # Generate AI response (your AI logic here)
        ai_response = "Machine learning is a subset of artificial intelligence..."
        
        # Validate output
        output_safe, output_result = await guardrails.check_output(
            ai_response, message
        )
        
        if output_safe:
            print("✅ Response is safe to send")
            print(f"AI: {ai_response}")
        else:
            print(f"❌ Response blocked: {output_result.reason}")
    else:
        # Get safety response
        safety_response = guardrails.get_safety_response(results)
        print(f"🛡️ Safety Response: {safety_response}")

asyncio.run(main())
```

## 📋 Configuration

### Rate Limiting Configuration

```python
# Configure rate limits
guardrails.rate_limiter.limits = {
    "messages_per_minute": 10,
    "messages_per_hour": 100,
    "warnings_per_session": 3,
    "session_timeout_minutes": 30
}
```

### Custom Safety Responses

```python
from guardrails_system import SafetyLevel

# Add custom safety responses
guardrails.add_custom_safety_response(
    SafetyLevel.BLOCKED,
    "I can't help with that request. Let's talk about something else."
)

guardrails.add_custom_safety_response(
    SafetyLevel.WARNING,
    "I want to be helpful while keeping our conversation appropriate."
)
```

### Custom Content Filters

```python
# Add custom harmful patterns
guardrails.content_filter.harmful_patterns.append(
    r'\b(?:custom|harmful|pattern)\b'
)

# Add custom PII patterns
guardrails.content_filter.pii_patterns['custom_id'] = r'\bCUST\d{6}\b'
```

## 🔄 Integration with FastAPI

The system includes a complete FastAPI integration example:

```python
from fastapi import FastAPI
from guardrails_system import GuardrailsSystem

app = FastAPI()

@app.post("/api/safe-chat")
async def safe_chat(request: ChatRequest):
    # Input validation
    input_safe, input_results = await guardrails.check_input(
        user_id=request.user_id,
        message=request.message,
        conversation_history=request.conversation_history
    )
    
    if not input_safe:
        return SafeChatResponse(
            response=guardrails.get_safety_response(input_results),
            sources=[],
            safety_info={"input_blocked": True},
            guardrails_triggered=[r.guardrail_type.value for r in input_results if not r.is_safe]
        )
    
    # Generate AI response and validate output
    # ... your AI logic here
    
    return SafeChatResponse(response=ai_response, sources=sources)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_guardrails.py
```

The test suite covers:
- Content filtering scenarios
- Rate limiting behavior
- Output validation
- Session management
- Toxicity detection
- Configuration management

## 📊 Safety Levels

| Level | Description | Action |
|-------|-------------|--------|
| `SAFE` | Content passes all checks | Proceed normally |
| `WARNING` | Minor issues detected | Proceed with disclaimer |
| `UNSAFE` | Significant safety concerns | Block with explanation |
| `BLOCKED` | Harmful content detected | Block completely |

## 🔍 Guardrail Types

| Type | Purpose | Examples |
|------|---------|----------|
| `CONTENT_FILTER` | Block harmful patterns | Violence, illegal activities |
| `TOXICITY_DETECTION` | AI-powered toxicity check | Hate speech, harassment |
| `PII_DETECTION` | Protect personal information | SSN, email, phone numbers |
| `RATE_LIMITING` | Prevent abuse | Message frequency limits |
| `CONTEXT_VALIDATION` | Prevent attacks | Prompt injection, context switching |
| `OUTPUT_VALIDATION` | Ensure safe responses | System leaks, relevance |

## 📈 Session Management

The system tracks user sessions to:
- Monitor message frequency
- Accumulate safety warnings
- Track violation history
- Implement cooling-off periods

```python
# Get session statistics
stats = guardrails.get_session_stats("user123")
print(f"Messages: {stats['message_count']}")
print(f"Warnings: {stats['warning_count']}")
print(f"Duration: {stats['session_duration']} seconds")

# Reset user session
if "user123" in guardrails.rate_limiter.user_sessions:
    del guardrails.rate_limiter.user_sessions["user123"]
```

## 🎯 Use Cases

### 1. **Customer Service Chatbots**
- Prevent inappropriate conversations
- Protect customer data (PII)
- Maintain professional interactions

### 2. **Educational AI Assistants**
- Block harmful content requests
- Handle sensitive topics appropriately
- Prevent misuse by students

### 3. **Healthcare AI**
- Detect medical advice requests
- Add appropriate disclaimers
- Protect patient information

### 4. **Financial AI**
- Block investment advice requests
- Protect financial information
- Prevent fraud attempts

## ⚠️ Important Considerations

### 1. **Performance Impact**
- Guardrails add latency (~100-500ms per request)
- OpenAI Moderation API calls have additional cost
- Consider caching for frequently checked content

### 2. **False Positives**
- Pattern-based filtering may have false positives
- Regularly review and update filter patterns
- Implement appeal mechanisms for users

### 3. **Customization**
- Adjust safety levels based on your use case
- Different applications need different safety thresholds
- Consider cultural and contextual factors

### 4. **Monitoring**
- Log all guardrail triggers for analysis
- Monitor false positive rates
- Track user experience impact

## 🔮 Advanced Features

### Custom Guardrail Implementation

```python
from guardrails_system import GuardrailResult, GuardrailType, SafetyLevel

class CustomGuardrail:
    def check_custom_rule(self, text: str) -> GuardrailResult:
        # Your custom logic here
        if "custom_violation" in text.lower():
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                guardrail_type=GuardrailType.CONTENT_FILTER,
                reason="Custom rule violation",
                confidence=0.9,
                suggested_action="Block and provide alternative"
            )
        
        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            guardrail_type=GuardrailType.CONTENT_FILTER,
            reason="Custom rule passed",
            confidence=0.8,
            suggested_action="Proceed normally"
        )
```

### Integration with Logging

```python
import logging

# Configure logging for guardrails
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardrails")

# Log guardrail events
if not input_safe:
    logger.warning(f"Input blocked for user {user_id}: {results[-1].reason}")
```

### Metrics and Analytics

```python
# Track guardrail metrics
metrics = {
    "total_requests": 0,
    "blocked_requests": 0,
    "warnings_issued": 0,
    "guardrail_triggers": {}
}

# Update metrics
for result in input_results:
    if not result.is_safe:
        metrics["guardrail_triggers"][result.guardrail_type.value] = \
            metrics["guardrail_triggers"].get(result.guardrail_type.value, 0) + 1
```

