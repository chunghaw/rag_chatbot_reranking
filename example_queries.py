#!/usr/bin/env python3
"""
Example Queries for YouTube Transcript RAG System

This script demonstrates how to effectively query the RAG system
with channel_name metadata for better search results.
"""

import requests
import json
from typing import List, Dict, Any, Optional

# Configuration
RAG_API_URL = "http://localhost:8000"


def chat_with_rag(message: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Send a chat message to the RAG system."""
    if conversation_history is None:
        conversation_history = []
    
    try:
        response = requests.post(
            f"{RAG_API_URL}/api/chat",
            json={
                "message": message,
                "conversation_history": conversation_history
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": f"Request failed: {e}"}


def print_chat_response(response: Dict[str, Any]):
    """Print a formatted chat response."""
    if "error" in response:
        print(f"❌ Error: {response['error']}")
        return
    
    print(f"\n🤖 AI Response:")
    print(f"{response.get('response', 'No response')}")
    
    sources = response.get('sources', [])
    if sources:
        print(f"\n📚 Sources ({len(sources)} found):")
        for i, source in enumerate(sources, 1):
            try:
                metadata = source.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                channel_name = metadata.get('channel_name', 'Unknown Channel')
                video_title = metadata.get('video_title', 'Unknown Video')
                youtube_id = metadata.get('youtube_id', '')
                start_time = metadata.get('start_time', 0)
                
                # Format timestamp
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                # Include YouTube ID if available
                youtube_info = f" (ID: {youtube_id})" if youtube_id else ""
                print(f"  {i}. {channel_name} - {video_title}{youtube_info} [{timestamp}]")
                print(f"     \"{source.get('text', '')[:100]}...\"")
                
            except Exception as e:
                print(f"  {i}. Error parsing source: {e}")
    
    print("-" * 80)


def run_example_queries():
    """Run example queries that demonstrate channel_name metadata usage."""
    
    print("🎯 YouTube Transcript RAG System - Example Queries")
    print("=" * 80)
    print("This demonstrates how channel_name metadata enhances search results")
    print("=" * 80)
    
    # Example queries that leverage channel_name metadata
    example_queries = [
        {
            "query": "What did JoeReisData say about data engineering best practices?",
            "description": "Channel-specific query for JoeReisData content"
        },
        {
            "query": "Show me insights about AI and machine learning from Matthew Berman",
            "description": "Channel-specific query for Matthew Berman content"
        },
        {
            "query": "What are the latest programming tips from Eczachly_?",
            "description": "Channel-specific query for Eczachly_ content"
        },
        {
            "query": "Find discussions about analytics and data science from Data With Danny",
            "description": "Channel-specific query for Data With Danny content"
        },
        {
            "query": "What do all creators say about Python programming?",
            "description": "Cross-channel query about a specific topic"
        },
        {
            "query": "Compare different approaches to data engineering across all channels",
            "description": "Comparative analysis across channels"
        },
        {
            "query": "What are the most recent insights about large language models?",
            "description": "Topic-based query that will show channel attribution"
        }
    ]
    
    conversation_history = []
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n🔍 Example {i}: {example['description']}")
        print(f"Query: \"{example['query']}\"")
        print("-" * 60)
        
        # Send query to RAG system
        response = chat_with_rag(example['query'], conversation_history)
        
        # Print response
        print_chat_response(response)
        
        # Add to conversation history for context
        if "error" not in response:
            conversation_history.append({
                "role": "user",
                "content": example['query']
            })
            conversation_history.append({
                "role": "assistant", 
                "content": response.get('response', '')
            })
        
        # Keep conversation history manageable
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
    
    print(f"\n🎉 Example queries completed!")
    print(f"💡 The channel_name metadata helps provide:")
    print(f"   • Source attribution for each response")
    print(f"   • Channel-specific filtering capabilities")
    print(f"   • Timestamp information for video segments")
    print(f"   • Better context for cross-channel comparisons")


def test_rag_connection():
    """Test if the RAG system is accessible."""
    print("🔍 Testing RAG system connection...")
    
    try:
        response = requests.get(f"{RAG_API_URL}/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ RAG system is running")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Milvus Connected: {health_data.get('milvus_connected', False)}")
            return True
        else:
            print(f"❌ RAG system returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to RAG system: {e}")
        print(f"💡 Make sure your RAG system is running on {RAG_API_URL}")
        return False


if __name__ == "__main__":
    print("🚀 YouTube Transcript RAG Query Examples")
    print("=" * 80)
    
    # Test connection first
    if not test_rag_connection():
        print("\n❌ Cannot proceed without RAG system connection")
        print("💡 Please start your RAG system first:")
        print("   python main.py")
        exit(1)
    
    # Run example queries
    run_example_queries() 