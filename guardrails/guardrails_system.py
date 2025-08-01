"""
AI Chatbot Guardrails System

This module implements comprehensive safety guardrails for AI chatbots including:
- Content filtering
- Safety classification
- Rate limiting
- Context awareness
- Output validation
- Toxicity detection
"""

import re
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import openai
from openai import AsyncOpenAI

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

class GuardrailType(Enum):
    CONTENT_FILTER = "content_filter"
    TOXICITY_DETECTION = "toxicity_detection"
    PII_DETECTION = "pii_detection"
    RATE_LIMITING = "rate_limiting"
    CONTEXT_VALIDATION = "context_validation"
    OUTPUT_VALIDATION = "output_validation"

@dataclass
class GuardrailResult:
    """Result of a guardrail check"""
    is_safe: bool
    safety_level: SafetyLevel
    guardrail_type: GuardrailType
    reason: str
    confidence: float
    suggested_action: str
    metadata: Dict[str, Any] = None

@dataclass
class UserSession:
    """User session tracking for rate limiting and context"""
    user_id: str
    session_start: datetime
    message_count: int
    last_message_time: datetime
    warning_count: int
    violation_history: List[str]

class ContentFilter:
    """Content filtering guardrail"""
    
    def __init__(self):
        # Harmful content patterns
        self.harmful_patterns = [
            r'\b(?:kill|murder|suicide|harm)\s+(?:yourself|myself|themselves)\b',
            r'\b(?:how\s+to\s+make|build|create)\s+(?:bomb|explosive|weapon)\b',
            r'\b(?:illegal\s+drug|drug\s+dealing|sell\s+drugs)\b',
            r'\b(?:hack|hacking|ddos|phishing|malware)\b',
            r'\b(?:child\s+abuse|sexual\s+content|explicit\s+content)\b',
        ]
        
        # Sensitive topics that need careful handling
        self.sensitive_patterns = [
            r'\b(?:depression|anxiety|mental\s+health|therapy)\b',
            r'\b(?:medical\s+advice|diagnosis|treatment)\b',
            r'\b(?:legal\s+advice|lawsuit|court)\b',
            r'\b(?:investment\s+advice|financial\s+advice|trading)\b',
        ]
        
        # PII patterns
        self.pii_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b\d{3}-?\d{3}-?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
    
    def check_content(self, text: str) -> GuardrailResult:
        """Check content for harmful patterns"""
        text_lower = text.lower()
        
        # Check for harmful content
        for pattern in self.harmful_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    guardrail_type=GuardrailType.CONTENT_FILTER,
                    reason=f"Detected harmful content pattern: {pattern}",
                    confidence=0.9,
                    suggested_action="Block request and provide safety resources",
                    metadata={"pattern_matched": pattern}
                )
        
        # Check for sensitive content
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=True,
                    safety_level=SafetyLevel.WARNING,
                    guardrail_type=GuardrailType.CONTENT_FILTER,
                    reason=f"Detected sensitive content: {pattern}",
                    confidence=0.7,
                    suggested_action="Proceed with disclaimer",
                    metadata={"pattern_matched": pattern, "requires_disclaimer": True}
                )
        
        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            guardrail_type=GuardrailType.CONTENT_FILTER,
            reason="No harmful content detected",
            confidence=0.8,
            suggested_action="Proceed normally"
        )
    
    def check_pii(self, text: str) -> GuardrailResult:
        """Check for personally identifiable information"""
        found_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                found_pii.append((pii_type, matches))
        
        if found_pii:
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.UNSAFE,
                guardrail_type=GuardrailType.PII_DETECTION,
                reason=f"Detected PII: {[pii[0] for pii in found_pii]}",
                confidence=0.95,
                suggested_action="Remove PII before processing",
                metadata={"pii_found": found_pii}
            )
        
        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            guardrail_type=GuardrailType.PII_DETECTION,
            reason="No PII detected",
            confidence=0.9,
            suggested_action="Proceed normally"
        )

class ToxicityDetector:
    """AI-powered toxicity detection using OpenAI moderation"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
    
    async def check_toxicity(self, text: str) -> GuardrailResult:
        """Use OpenAI moderation API to detect toxicity"""
        try:
            response = await self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Get categories and scores
            categories = result.categories
            category_scores = result.category_scores
            
            if result.flagged:
                # Get the highest scoring category
                flagged_categories = []
                max_score = 0.0
                max_category = "unknown"
                print(categories)
                print(category_scores)
                
                # Safely iterate through categories
                for category_name in dir(categories):
                    if not category_name.startswith('_'):
                        try:
                            is_flagged = getattr(categories, category_name, False)
                            score = getattr(category_scores, category_name, 0.0)
                            
                            if is_flagged:
                                flagged_categories.append(category_name)
                            
                            if score > max_score:
                                max_score = score
                                max_category = category_name
                                
                        except (AttributeError, TypeError):
                            continue
                
                return GuardrailResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    guardrail_type=GuardrailType.TOXICITY_DETECTION,
                    reason=f"Content flagged for: {', '.join(flagged_categories) if flagged_categories else 'policy violation'}",
                    confidence=max_score,
                    suggested_action="Block content and provide alternative response",
                    metadata={
                        "flagged_categories": flagged_categories,
                        "max_score": max_score,
                        "max_category": max_category
                    }
                )
            
            # Calculate confidence for safe content
            max_score = 0.0
            try:
                for category_name in dir(category_scores):
                    if not category_name.startswith('_'):
                        try:
                            score = getattr(category_scores, category_name, 0.0)
                            if score > max_score:
                                max_score = score
                        except (AttributeError, TypeError):
                            continue
            except Exception:
                max_score = 0.1  # Default low score if we can't read scores
            
            return GuardrailResult(
                is_safe=True,
                safety_level=SafetyLevel.SAFE,
                guardrail_type=GuardrailType.TOXICITY_DETECTION,
                reason="Content passed toxicity check",
                confidence=1.0 - max_score,
                suggested_action="Proceed normally"
            )
            
        except Exception as e:
            # Fallback in case of API issues
            return GuardrailResult(
                is_safe=True,
                safety_level=SafetyLevel.WARNING,
                guardrail_type=GuardrailType.TOXICITY_DETECTION,
                reason=f"Toxicity check failed: {str(e)}",
                confidence=0.5,
                suggested_action="Proceed with caution",
                metadata={"error": str(e)}
            )

class RateLimiter:
    """Rate limiting guardrail"""
    
    def __init__(self):
        self.user_sessions: Dict[str, UserSession] = {}
        
        # Rate limiting rules
        self.limits = {
            "messages_per_minute": 10,
            "messages_per_hour": 100,
            "warnings_per_session": 3,
            "session_timeout_minutes": 30
        }
    
    def check_rate_limit(self, user_id: str) -> GuardrailResult:
        """Check if user exceeds rate limits"""
        now = datetime.now()
        
        # Get or create user session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = UserSession(
                user_id=user_id,
                session_start=now,
                message_count=0,
                last_message_time=now,
                warning_count=0,
                violation_history=[]
            )
        
        session = self.user_sessions[user_id]
        
        # Check session timeout
        if now - session.last_message_time > timedelta(minutes=self.limits["session_timeout_minutes"]):
            # Reset session
            session.message_count = 0
            session.warning_count = 0
            session.session_start = now
        
        # Check messages per minute
        if now - session.last_message_time < timedelta(minutes=1):
            if session.message_count >= self.limits["messages_per_minute"]:
                return GuardrailResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    guardrail_type=GuardrailType.RATE_LIMITING,
                    reason="Rate limit exceeded: too many messages per minute",
                    confidence=1.0,
                    suggested_action="Block request and ask user to slow down",
                    metadata={"limit_type": "per_minute", "count": session.message_count}
                )
        
        # Check messages per hour
        if now - session.session_start < timedelta(hours=1):
            if session.message_count >= self.limits["messages_per_hour"]:
                return GuardrailResult(
                    is_safe=False,
                    safety_level=SafetyLevel.BLOCKED,
                    guardrail_type=GuardrailType.RATE_LIMITING,
                    reason="Rate limit exceeded: too many messages per hour",
                    confidence=1.0,
                    suggested_action="Block request and implement cooling off period",
                    metadata={"limit_type": "per_hour", "count": session.message_count}
                )
        
        # Check warning accumulation
        if session.warning_count >= self.limits["warnings_per_session"]:
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                guardrail_type=GuardrailType.RATE_LIMITING,
                reason="Too many warnings in session",
                confidence=1.0,
                suggested_action="End session and require fresh start",
                metadata={"warning_count": session.warning_count}
            )
        
        # Update session
        session.message_count += 1
        session.last_message_time = now
        
        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            guardrail_type=GuardrailType.RATE_LIMITING,
            reason="Within rate limits",
            confidence=1.0,
            suggested_action="Proceed normally",
            metadata={"message_count": session.message_count}
        )
    
    def add_warning(self, user_id: str, reason: str):
        """Add a warning to user session"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            session.warning_count += 1
            session.violation_history.append(f"{datetime.now().isoformat()}: {reason}")

class ContextValidator:
    """Context and conversation validation"""
    
    def __init__(self):
        self.max_context_length = 4000
        self.max_turns = 20
        
    def validate_context(self, conversation_history: List[Dict], current_message: str) -> GuardrailResult:
        """Validate conversation context"""
        
        # Check conversation length
        if len(conversation_history) > self.max_turns:
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                guardrail_type=GuardrailType.CONTEXT_VALIDATION,
                reason="Conversation too long, context may be degraded",
                confidence=0.8,
                suggested_action="Truncate conversation history",
                metadata={"turn_count": len(conversation_history)}
            )
        
        # Check total context size
        total_context = " ".join([msg.get("content", "") for msg in conversation_history]) + current_message
        if len(total_context) > self.max_context_length:
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                guardrail_type=GuardrailType.CONTEXT_VALIDATION,
                reason="Context length exceeds maximum",
                confidence=0.9,
                suggested_action="Truncate context to fit limits",
                metadata={"context_length": len(total_context)}
            )
        
        # Check for context switching attacks
        if self._detect_context_switching(conversation_history, current_message):
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.UNSAFE,
                guardrail_type=GuardrailType.CONTEXT_VALIDATION,
                reason="Potential context switching attack detected",
                confidence=0.7,
                suggested_action="Reset conversation context",
                metadata={"attack_type": "context_switching"}
            )
        
        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            guardrail_type=GuardrailType.CONTEXT_VALIDATION,
            reason="Context validation passed",
            confidence=0.9,
            suggested_action="Proceed normally"
        )
    
    def _detect_context_switching(self, history: List[Dict], message: str) -> bool:
        """Detect potential context switching attacks"""
        switching_indicators = [
            "ignore previous instructions",
            "forget what i said before",
            "new instructions:",
            "system prompt:",
            "you are now",
            "pretend you are",
            "act as if",
            "roleplay as"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in switching_indicators)

class OutputValidator:
    """Validate AI responses before sending to user"""
    
    def __init__(self):
        self.forbidden_outputs = [
            "I cannot help with that",
            "I'm not allowed to",
            "That violates my guidelines"
        ]
    
    def validate_output(self, response: str, original_query: str) -> GuardrailResult:
        """Validate AI-generated response"""
        
        # Check for leaked system prompts
        if self._contains_system_leak(response):
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.BLOCKED,
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                reason="Response contains leaked system information",
                confidence=0.9,
                suggested_action="Generate new response",
                metadata={"leak_type": "system_prompt"}
            )
        
        # Check response relevance
        if not self._is_relevant_response(response, original_query):
            return GuardrailResult(
                is_safe=False,
                safety_level=SafetyLevel.WARNING,
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                reason="Response not relevant to query",
                confidence=0.6,
                suggested_action="Generate more relevant response",
                metadata={"relevance_score": 0.3}
            )
        
        # Check response length
        if len(response) > 2000:
            return GuardrailResult(
                is_safe=True,
                safety_level=SafetyLevel.WARNING,
                guardrail_type=GuardrailType.OUTPUT_VALIDATION,
                reason="Response very long",
                confidence=0.8,
                suggested_action="Consider truncating response",
                metadata={"response_length": len(response)}
            )
        
        return GuardrailResult(
            is_safe=True,
            safety_level=SafetyLevel.SAFE,
            guardrail_type=GuardrailType.OUTPUT_VALIDATION,
            reason="Output validation passed",
            confidence=0.9,
            suggested_action="Send response to user"
        )
    
    def _contains_system_leak(self, response: str) -> bool:
        """Check if response contains system prompt leaks"""
        leak_indicators = [
            "system:",
            "assistant:",
            "you are an ai",
            "your instructions are",
            "your role is to",
            "you must not",
            "you are programmed to"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in leak_indicators)
    
    def _is_relevant_response(self, response: str, query: str) -> bool:
        """Basic relevance check"""
        # Simple keyword overlap check
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(query_words.intersection(response_words))
        return overlap >= max(1, len(query_words) * 0.2)

class GuardrailsSystem:
    """Main guardrails system orchestrator"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.content_filter = ContentFilter()
        self.toxicity_detector = ToxicityDetector(openai_client)
        self.rate_limiter = RateLimiter()
        self.context_validator = ContextValidator()
        self.output_validator = OutputValidator()
        
        # Safety responses
        self.safety_responses = {
            SafetyLevel.BLOCKED: "I can't help with that request. Let's talk about something else.",
            SafetyLevel.UNSAFE: "I noticed some concerning content. Let me help you with something safer.",
            SafetyLevel.WARNING: "I want to be helpful while keeping our conversation appropriate."
        }
    
    async def check_input(self, user_id: str, message: str, conversation_history: List[Dict]) -> Tuple[bool, List[GuardrailResult]]:
        """Run all input guardrails"""
        results = []
        
        # 1. Rate limiting
        rate_result = self.rate_limiter.check_rate_limit(user_id)
        results.append(rate_result)
        if not rate_result.is_safe:
            return False, results
        
        # 2. Content filtering
        content_result = self.content_filter.check_content(message)
        results.append(content_result)
        if not content_result.is_safe:
            return False, results
        
        # 3. PII detection
        pii_result = self.content_filter.check_pii(message)
        results.append(pii_result)
        if not pii_result.is_safe:
            return False, results
        
        # 4. Toxicity detection
        toxicity_result = await self.toxicity_detector.check_toxicity(message)
        results.append(toxicity_result)
        if not toxicity_result.is_safe:
            return False, results
        
        # 5. Context validation
        context_result = self.context_validator.validate_context(conversation_history, message)
        results.append(context_result)
        if not context_result.is_safe and context_result.safety_level == SafetyLevel.BLOCKED:
            return False, results
        
        # Add warnings to user session
        for result in results:
            if result.safety_level in [SafetyLevel.WARNING, SafetyLevel.UNSAFE]:
                self.rate_limiter.add_warning(user_id, result.reason)
        
        return True, results
    
    async def check_output(self, response: str, original_query: str) -> Tuple[bool, GuardrailResult]:
        """Run output guardrails"""
        
        # 1. Output validation
        output_result = self.output_validator.validate_output(response, original_query)
        if not output_result.is_safe:
            return False, output_result
        
        # 2. Re-run toxicity check on output
        toxicity_result = await self.toxicity_detector.check_toxicity(response)
        if not toxicity_result.is_safe:
            return False, toxicity_result
        
        return True, output_result
    
    def get_safety_response(self, results: List[GuardrailResult]) -> str:
        """Get appropriate safety response based on guardrail results"""
        
        # Find the most severe safety level
        max_severity = SafetyLevel.SAFE
        for result in results:
            if result.safety_level == SafetyLevel.BLOCKED:
                max_severity = SafetyLevel.BLOCKED
                break
            elif result.safety_level == SafetyLevel.UNSAFE:
                max_severity = SafetyLevel.UNSAFE
            elif result.safety_level == SafetyLevel.WARNING and max_severity == SafetyLevel.SAFE:
                max_severity = SafetyLevel.WARNING
        
        if max_severity in self.safety_responses:
            return self.safety_responses[max_severity]
        
        return "I'm here to help with appropriate questions. What else can I assist you with?"
    
    def add_custom_safety_response(self, level: SafetyLevel, response: str):
        """Add custom safety response"""
        self.safety_responses[level] = response
    
    def get_session_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user session statistics"""
        if user_id in self.rate_limiter.user_sessions:
            session = self.rate_limiter.user_sessions[user_id]
            return {
                "message_count": session.message_count,
                "warning_count": session.warning_count,
                "session_duration": (datetime.now() - session.session_start).total_seconds(),
                "violation_history": session.violation_history
            }
        return {"message_count": 0, "warning_count": 0, "session_duration": 0, "violation_history": []} 