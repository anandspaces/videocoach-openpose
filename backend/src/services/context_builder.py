"""
Context Builder - Builds rich context for Gemini API
Combines pose data, speech transcript, conversation history, and error patterns
"""

from typing import Dict, Any, List, Optional
from collections import deque


class ContextBuilder:
    """
    Builds comprehensive context for Gemini API coaching requests
    Maintains conversation history and error patterns
    """
    
    def __init__(
        self,
        max_conversation_history: int = 3,
        max_error_history: int = 10
    ):
        """
        Initialize context builder
        
        Args:
            max_conversation_history: Number of recent exchanges to keep (default: 3)
            max_error_history: Number of recent errors to track (default: 10)
        """
        self.conversation_history = deque(maxlen=max_conversation_history)
        self.error_history = deque(maxlen=max_error_history)
    
    def build_context(
        self,
        transcript: str,
        pose_snapshot: Dict[str, Any],
        asana_definition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build complete context for Gemini API
        
        Args:
            transcript: User's speech transcript
            pose_snapshot: Current pose analysis snapshot
            asana_definition: Optional asana definition/rules
            
        Returns:
            Dictionary with complete context for Gemini
        """
        context = {
            'transcript': transcript,
            'pose_snapshot': {
                'current_angles': pose_snapshot.get('angles', {}),
                'detected_errors': pose_snapshot.get('errors', []),
                'stability_score': pose_snapshot.get('stability', 0.0),
                'asana_progress': pose_snapshot.get('completion_percentage', 0.0),
                'balance_score': pose_snapshot.get('balance_score', 0.0),
                'posture_status': pose_snapshot.get('posture_status', 'Unknown')
            },
            'conversation_history': list(self.conversation_history),
            'error_history': list(self.error_history)
        }
        
        # Add asana definition if provided
        if asana_definition:
            context['asana_definition'] = asana_definition
        
        return context
    
    def build_prompt(
        self,
        context: Dict[str, Any],
        asana_name: str = "yoga pose"
    ) -> str:
        """
        Build formatted prompt for Gemini API
        
        Args:
            context: Context dictionary from build_context()
            asana_name: Name of the asana being performed
            
        Returns:
            Formatted prompt string
        """
        transcript = context.get('transcript', '')
        pose = context.get('pose_snapshot', {})
        
        # Build pose analysis summary
        pose_summary = []
        
        # Angles
        angles = pose.get('current_angles', {})
        if angles:
            pose_summary.append("Current joint angles:")
            for joint, angle in angles.items():
                pose_summary.append(f"  - {joint}: {angle:.1f}°")
        
        # Errors
        errors = pose.get('detected_errors', [])
        if errors:
            pose_summary.append("\nDetected issues:")
            for error in errors[:3]:  # Limit to top 3 errors
                if isinstance(error, dict):
                    msg = error.get('message', str(error))
                else:
                    msg = str(error)
                pose_summary.append(f"  - {msg}")
        
        # Stability and progress
        stability = pose.get('stability_score', 0.0)
        progress = pose.get('asana_progress', 0.0)
        balance = pose.get('balance_score', 0.0)
        posture = pose.get('posture_status', 'Unknown')
        
        pose_summary.append(f"\nPerformance metrics:")
        pose_summary.append(f"  - Stability: {stability:.1f}%")
        pose_summary.append(f"  - Balance: {balance:.1f}/100")
        pose_summary.append(f"  - Posture: {posture}")
        pose_summary.append(f"  - Overall progress: {progress:.1f}%")
        
        pose_text = "\n".join(pose_summary)
        
        # Build conversation context
        history = context.get('conversation_history', [])
        history_text = ""
        if history:
            history_text = "\n\nRecent conversation:\n"
            for exchange in history[-2:]:  # Last 2 exchanges
                user_msg = exchange.get('user', '')
                ai_msg = exchange.get('ai', '')
                if user_msg:
                    history_text += f"User: {user_msg}\n"
                if ai_msg:
                    history_text += f"Coach: {ai_msg}\n"
        
        # Build final prompt
        prompt = f"""You are a real-time yoga coach. The user is performing {asana_name}.

User said: "{transcript}"

Current pose analysis:
{pose_text}{history_text}

Provide a brief, encouraging response with ONE specific correction or affirmation. Keep it conversational and under 150 tokens. Focus on what the user asked about."""
        
        return prompt
    
    def add_exchange(
        self,
        user_message: str,
        ai_response: str
    ) -> None:
        """
        Add a conversation exchange to history
        
        Args:
            user_message: User's message
            ai_response: AI's response
        """
        self.conversation_history.append({
            'user': user_message,
            'ai': ai_response,
            'timestamp': self._get_timestamp()
        })
    
    def add_error(
        self,
        error: Dict[str, Any]
    ) -> None:
        """
        Add an error to error history
        
        Args:
            error: Error dictionary
        """
        error_entry = {
            'error': error,
            'timestamp': self._get_timestamp()
        }
        self.error_history.append(error_entry)
    
    def get_recent_errors(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent errors
        
        Args:
            count: Number of errors to retrieve
            
        Returns:
            List of recent errors
        """
        return list(self.error_history)[-count:]
    
    def clear_history(self) -> None:
        """Clear conversation and error history"""
        self.conversation_history.clear()
        self.error_history.clear()
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context builder statistics
        
        Returns:
            Dictionary with stats
        """
        return {
            'conversation_exchanges': len(self.conversation_history),
            'errors_tracked': len(self.error_history),
            'last_user_message': self.conversation_history[-1].get('user', '') if self.conversation_history else '',
            'last_ai_response': self.conversation_history[-1].get('ai', '') if self.conversation_history else ''
        }
