#!/usr/bin/env python3
"""
Guardrails AI Hub - Multi-Validator Example
===========================================

This example demonstrates how to use multiple validators from the Guardrails Hub
for comprehensive AI safety including PII detection and toxic language filtering.

Based on: https://hub.guardrailsai.com/

Features demonstrated:
1. PII validator for personal information detection
2. ToxicLanguage validator for harmful content detection
3. Multiple validator composition
4. Professional validation patterns
5. Comprehensive safety testing
"""

import os
import asyncio
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Basic imports
from openai import AsyncOpenAI
from dotenv import load_dotenv


# Official Guardrails Hub imports
from guardrails.hub import DetectPII, ToxicLanguage
from guardrails import Guard



# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_safe: bool
    validator_name: str
    confidence: float
    reason: str
    suggested_action: str
    metadata: Dict[str, Any] = None


class GuardrailsHubDemo:
    """Demonstration of Guardrails Hub multi-validator system"""
    
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self.setup_guards()
    
    def setup_guards(self):
        """Setup guards with Hub validators"""
        # PII Detection Guard
        self.pii_guard = Guard().use(
            DetectPII, 
            pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD", "US_PASSPORT"],
            redact=True,
            on_fail="fix"
        )
        
        # Toxic Language Guard
        self.toxicity_guard = Guard().use(
            ToxicLanguage, 
            threshold=0.5, 
            validation_method="sentence", 
            on_fail="exception"
        )
        
        # Combined Guard with multiple validators
        self.combined_guard = Guard().use_many(
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"],
                redact=True,
                on_fail="fix"
            ),
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence", 
                on_fail="exception"
            )
        )
        
        print("✅ Using Guardrails Hub: PII + ToxicLanguage validators")
    
    async def validate_with_hub(self, text: str, validator_type: str = "combined") -> ValidationResult:
        """Validate text using Guardrails Hub validators"""
        try:
            if validator_type == "pii":
                # Test with PII validator only
                validated_text = self.pii_guard.validate(text)
                pii_detected = text != validated_text
                
                return ValidationResult(
                    is_safe=not pii_detected,
                    validator_name="PII",
                    confidence=0.95,
                    reason="PII detected and redacted" if pii_detected else "No PII detected in text",
                    suggested_action="Use redacted version" if pii_detected else "Proceed with original text",
                    metadata={"original_text": text, "redacted_text": validated_text}
                )
                
            elif validator_type == "toxicity":
                # Test with ToxicLanguage validator only
                validated_text = self.toxicity_guard.validate(text)
                
                return ValidationResult(
                    is_safe=True,
                    validator_name="ToxicLanguage",
                    confidence=0.95,
                    reason="Text passed toxicity validation",
                    suggested_action="Proceed with response",
                    metadata={"original_text": text, "validated_text": validated_text}
                )
                
            else:  # combined
                # Test with both validators
                validated_text = self.combined_guard.validate(text)
                pii_detected = text != validated_text
                
                return ValidationResult(
                    is_safe=not pii_detected,
                    validator_name="PII + ToxicLanguage",
                    confidence=0.95,
                    reason="PII detected and redacted" if pii_detected else "All validations passed",
                    suggested_action="Use redacted version" if pii_detected else "Proceed with original text",
                    metadata={"original_text": text, "redacted_text": validated_text}
                )
            
        except Exception as e:
            # Determine which validator failed based on error message
            validator_name = "ToxicLanguage" if "toxic" in str(e).lower() else validator_type.upper()
            
            return ValidationResult(
                is_safe=False,
                validator_name=validator_name,
                confidence=0.90,
                reason=str(e),
                suggested_action="Block content and provide alternative response",
                metadata={"error": str(e), "original_text": text}
            )
    
    async def generate_safe_response(self, user_input: str) -> Dict[str, Any]:
        """Generate AI response with Hub validation"""
        start_time = datetime.now()
        
        # Step 1: Validate input using Hub validators
        input_validation = await self.validate_with_hub(user_input)
        
        if not input_validation.is_safe:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            redacted_text = input_validation.metadata.get("redacted_text", user_input) if input_validation.metadata else user_input
            return {
                "response": f"I've detected personal information in your message. Here's the redacted version: {redacted_text}",
                "validation_info": {
                    "input_safe": False,
                    "validator_used": input_validation.validator_name,
                    "reason": input_validation.reason,
                    "confidence": input_validation.confidence,
                    "pii_redacted": True
                },
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: Generate AI response
        if self.client:
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful and respectful AI assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content
            except Exception as e:
                ai_response = "I'm having trouble generating a response right now. Please try again later."
        else:
            ai_response = f"Mock response for: {user_input[:50]}... (OpenAI not configured)"
        
        # Step 3: Validate output
        output_validation = await self.validate_with_hub(ai_response)
        
        if not output_validation.is_safe:
            # Use the redacted version if PII was detected
            if output_validation.metadata and "redacted_text" in output_validation.metadata:
                ai_response = output_validation.metadata["redacted_text"]
            else:
                ai_response = "I need to provide a different response to protect personal information."
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "response": ai_response,
            "validation_info": {
                "input_safe": input_validation.is_safe,
                "output_safe": output_validation.is_safe,
                "validator_used": "PII",
                "input_confidence": input_validation.confidence,
                "output_confidence": output_validation.confidence,
                "pii_detected_input": not input_validation.is_safe,
                "pii_detected_output": not output_validation.is_safe
            },
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat()
        }

async def demo_guardrails_hub():
    """Demonstrate Guardrails Hub multi-validator system"""
    
    print("🛡️ Guardrails Hub - Multi-Validator Demo")
    print("=" * 60)
    print("🌐 Hub: https://hub.guardrailsai.com/")
    print()
    
    # Initialize demo
    demo = GuardrailsHubDemo(OPENAI_API_KEY)
    
    # Test cases for PII detection
    pii_test_cases = [
        {
            "name": "Clean Text (No PII)",
            "input": "I love working on machine learning projects and data science.",
            "expected": "safe"
        },
        {
            "name": "Email Address",
            "input": "Please contact me at john.doe@example.com for more information.",
            "expected": "pii_detected"
        },
        {
            "name": "Phone Number",
            "input": "You can reach me at (555) 123-4567 or my office number.",
            "expected": "pii_detected"
        },
        {
            "name": "Credit Card Number",
            "input": "Use credit card 4532-1234-5678-9012 for the payment.",
            "expected": "pii_detected"
        }
    ]
    
    # Test cases for toxic language detection
    toxicity_test_cases = [
        {
            "name": "Positive Content",
            "input": "Love how you think and attack the problem. Great job!",
            "expected": "safe"
        },
        {
            "name": "Toxic Language",
            "input": "You are a stupid idiot who can't do anything right.",
            "expected": "blocked"
        },
        {
            "name": "Educational Query",
            "input": "Can you explain machine learning algorithms?",
            "expected": "safe"
        },
        {
            "name": "Hostile Language",
            "input": "I hate this stupid system and everyone who made it.",
            "expected": "blocked"
        }
    ]
    
    # Combined test cases
    combined_test_cases = [
        {
            "name": "Clean & Safe",
            "input": "I appreciate your help with this project.",
            "expected": "safe"
        },
        {
            "name": "PII + Toxic",
            "input": "You stupid fool, my email is john@email.com and you better respond!",
            "expected": "blocked_or_redacted"
        },
        {
            "name": "PII Only",
            "input": "Please call me at (555) 123-4567 when you get a chance.",
            "expected": "pii_detected"
        },
        {
            "name": "Toxic Only",
            "input": "This is absolutely terrible and worthless garbage.",
            "expected": "blocked"
        }
    ]
    
    # Test PII Detection
    print("🔍 Testing PII Detection Validator")
    print("-" * 50)
    
    for i, test_case in enumerate(pii_test_cases, 1):
        print(f"\nPII Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        validation_result = await demo.validate_with_hub(test_case['input'], "pii")
        
        status = "✅ NO PII" if validation_result.is_safe else "🔍 PII DETECTED"
        print(f"Result: {status}")
        print(f"Validator: {validation_result.validator_name}")
        print(f"Reason: {validation_result.reason}")
        
        if not validation_result.is_safe and validation_result.metadata:
            print(f"Redacted: {validation_result.metadata.get('redacted_text', '')}")
    
    # Test Toxic Language Detection
    print(f"\n{'='*60}")
    print("🚫 Testing Toxic Language Validator")
    print("-" * 50)
    
    for i, test_case in enumerate(toxicity_test_cases, 1):
        print(f"\nToxicity Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        validation_result = await demo.validate_with_hub(test_case['input'], "toxicity")
        
        status = "✅ SAFE" if validation_result.is_safe else "❌ BLOCKED"
        print(f"Result: {status}")
        print(f"Validator: {validation_result.validator_name}")
        print(f"Reason: {validation_result.reason}")
    
    # Test Combined Validation
    print(f"\n{'='*60}")
    print("🛡️ Testing Combined Validators")
    print("-" * 50)
    
    for i, test_case in enumerate(combined_test_cases, 1):
        print(f"\nCombined Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        validation_result = await demo.validate_with_hub(test_case['input'], "combined")
        
        if validation_result.is_safe:
            status = "✅ ALL CHECKS PASSED"
        else:
            status = "⚠️ FLAGGED" if "redacted" in validation_result.reason.lower() else "❌ BLOCKED"
        
        print(f"Result: {status}")
        print(f"Validators: {validation_result.validator_name}")
        print(f"Reason: {validation_result.reason}")
        
        if not validation_result.is_safe and validation_result.metadata:
            original = validation_result.metadata.get('original_text', '')
            redacted = validation_result.metadata.get('redacted_text', '')
            if original != redacted:
                print(f"Redacted: {redacted}")
    
    print("\n" + "="*60)
    print("🤖 Testing Complete Response Generation")
    print("-" * 50)
    
    safe_queries = [
        "What are best practices for data privacy in applications?",
        "How can I implement secure authentication without storing personal data?",
        "Explain GDPR compliance for data collection systems"
    ]
    
    for i, query in enumerate(safe_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        response_data = await demo.generate_safe_response(query)
        
        print(f"Response: {response_data['response'][:80]}...")
        print(f"Validation: {response_data['validation_info']}")
        print(f"Processing Time: {response_data['processing_time_ms']:.1f}ms")
    
    print("\n" + "="*60)
    print("🏆 Guardrails Hub Benefits")
    print("-" * 50)
    
    print("✅ Professional Validators:")
    print("  • PII - Identifies personally identifiable information")
    print("  • ToxicLanguage - Detects harmful and toxic content")
    print("  • Profanity - Filters inappropriate language")
    print("  • PromptInjection - Prevents prompt injection attacks")
    print("  • And many more at https://hub.guardrailsai.com/")
    
    print("\n🔧 Key Advantages:")
    print("  • Pre-built and tested validators")
    print("  • Regular updates and improvements") 
    print("  • Professional support and documentation")
    print("  • Easy integration with existing code")
    print("  • Configurable thresholds and behaviors")
    
    print("\n📚 Integration Example:")
    print("""
# Import from Guardrails Hub
from guardrails.hub import DetectPII, ToxicLanguage
from guardrails import Guard

# Option 1: Individual validators
pii_guard = Guard().use(DetectPII, pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], redact=True)
toxicity_guard = Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence")

# Option 2: Combined validators
combined_guard = Guard().use_many(
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], redact=True, on_fail="fix"),
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception")
)

# Test examples
result1 = combined_guard.validate("I love working on data science!")  # Safe
result2 = combined_guard.validate("Call me at (555) 123-4567")  # PII redacted
result3 = combined_guard.validate("You are terrible!")  # Toxic blocked
    """)

async def test_specific_examples():
    """Test specific examples for multi-validator detection"""
    print("\n🎯 Testing Specific Multi-Validator Examples")
    print("=" * 50)
    
    demo = GuardrailsHubDemo(OPENAI_API_KEY)
    
    # Test 1: Clean and safe text
    print("\nTest 1: Clean & Safe Content")
    text1 = "I'm interested in learning more about your data science services."
    print(f"Input: {text1}")
    
    result1 = await demo.validate_with_hub(text1, "combined")
    print(f"Result: {'✅ ALL SAFE' if result1.is_safe else '⚠️ FLAGGED'}")
    print(f"Validators: {result1.validator_name}")
    print(f"Reason: {result1.reason}")
    
    # Test 2: PII detection
    print("\nTest 2: Content with PII")
    text2 = "Please send the invoice to john.smith@company.com and call me at (555) 987-6543."
    print(f"Input: {text2}")
    
    result2 = await demo.validate_with_hub(text2, "combined")
    print(f"Result: {'✅ SAFE' if result2.is_safe else '🔍 PII DETECTED'}")
    print(f"Validators: {result2.validator_name}")
    print(f"Reason: {result2.reason}")
    if result2.metadata:
        print(f"Redacted: {result2.metadata.get('redacted_text', 'N/A')}")
    
    # Test 3: Toxic language detection
    print("\nTest 3: Toxic Content")
    text3 = "You are absolutely worthless and stupid!"
    print(f"Input: {text3}")
    
    result3 = await demo.validate_with_hub(text3, "combined")
    print(f"Result: {'✅ SAFE' if result3.is_safe else '❌ BLOCKED'}")
    print(f"Validators: {result3.validator_name}")
    print(f"Reason: {result3.reason}")
    
    # Test 4: PII + Toxic combined
    print("\nTest 4: PII + Toxic Content")
    text4 = "You idiot! Contact me at bad@email.com right now!"
    print(f"Input: {text4}")
    
    result4 = await demo.validate_with_hub(text4, "combined")
    print(f"Result: {'✅ SAFE' if result4.is_safe else '❌ BLOCKED/FLAGGED'}")
    print(f"Validators: {result4.validator_name}")
    print(f"Reason: {result4.reason}")
    if result4.metadata:
        redacted = result4.metadata.get('redacted_text', 'N/A')
        if redacted != 'N/A':
            print(f"Redacted: {redacted}")

if __name__ == "__main__":
    print("🚀 Starting Guardrails Hub Multi-Validator Demo...")
    print("   📋 Testing: PII Detection + Toxic Language Detection")
    
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY not found - using mock responses")
    else:
        print("✅ OpenAI API key found")
    
    print()
    asyncio.run(demo_guardrails_hub())
    asyncio.run(test_specific_examples()) 