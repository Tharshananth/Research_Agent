

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Try to import Together, but handle gracefully if not available
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    print("Warning: Together library not installed. Run: pip install together")
    Together = None
    TOGETHER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available model types for different tasks"""
    LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

@dataclass
class LLMResponse:
    """Structure for LLM responses"""
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class LLMManager:
    """
    Central manager for all LLM operations using Together API
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not TOGETHER_AVAILABLE:
            logger.warning("Together API not available - running in mock mode")
            self.client = None
            self.mock_mode = True
        else:
            if not self.api_key:
                logger.warning("No Together API key provided - running in mock mode")
                self.client = None
                self.mock_mode = True
            else:
                try:
                    self.client = Together(api_key=self.api_key)
                    self.mock_mode = False
                    logger.info("Together API client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Together client: {e}")
                    self.client = None
                    self.mock_mode = True

        # Model configurations
        self.model_configs = {
            ModelType.LLAMA_70B: {
                "name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "best_for": ["analysis", "reasoning", "complex_tasks"]
            },
        }

        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0

        # Default model for different tasks
        self.default_models = {
            "extraction": ModelType.LLAMA_70B,
            "summarization": ModelType.LLAMA_70B,
            "classification": ModelType.LLAMA_70B,
            "analysis": ModelType.LLAMA_70B
        }

    def get_response(self, prompt: str, model_type: ModelType = ModelType.LLAMA_70B, json_mode: bool = False) -> str:
        """
        Generates a response from the selected LLM using the prompt and model type.
        """
        if self.mock_mode:
            return self._get_mock_response(prompt, json_mode)
        
        self._wait_for_rate_limit()

        model_config = self.model_configs[model_type]
        model_name = model_config["name"]

        if json_mode:
            # Ensure the prompt explicitly requests JSON output
            system_prompt = "You are a helpful assistant. Respond only in JSON format."
            prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                top_p=model_config["top_p"],
            )
            self.request_count += 1
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return self._get_mock_response(prompt, json_mode)

    def _get_mock_response(self, prompt: str, json_mode: bool = False) -> str:
        """Generate mock response for testing"""
        if json_mode:
            return json.dumps({
                "key_contributions": ["Mock contribution 1", "Mock contribution 2"],
                "main_findings": ["Mock finding 1", "Mock finding 2"],
                "technical_approach": "Mock technical approach",
                "research_gap": "Mock research gap"
            })
        else:
            return "Mock response: This is a simulated response for testing purposes."

    def _wait_for_rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _format_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict]:
        """Format prompt into messages for Together API"""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def generate_response(self,
                         prompt: str,
                         model: Optional[ModelType] = None,
                         system_prompt: Optional[str] = None,
                         task_type: str = "analysis",
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         json_mode: bool = False) -> str:
        """Generate response using Together API"""
        try:
            response = self.generate_response_detailed(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                task_type=task_type,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode
            )
            return response.content if response.success else ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_mock_response(prompt, json_mode)

    def generate_response_detailed(self,
                                  prompt: str,
                                  model: Optional[ModelType] = None,
                                  system_prompt: Optional[str] = None,
                                  task_type: str = "analysis",
                                  max_tokens: Optional[int] = None,
                                  temperature: Optional[float] = None,
                                  json_mode: bool = False) -> LLMResponse:
        """Generate detailed response with metadata"""

        # Select model
        if model is None:
            model = self.default_models.get(task_type, ModelType.LLAMA_70B)

        model_config = self.model_configs[model]

        if self.mock_mode:
            return LLMResponse(
                content=self._get_mock_response(prompt, json_mode),
                model=model.value,
                tokens_used=100,
                response_time=0.5,
                success=True
            )

        # Prepare parameters
        params = {
            "model": model_config["name"],
            "messages": self._format_messages(prompt, system_prompt),
            "max_tokens": max_tokens or model_config["max_tokens"],
            "temperature": temperature or model_config["temperature"],
            "top_p": model_config["top_p"]
        }

        # Add JSON mode instruction if requested
        if json_mode and system_prompt is None:
            params["messages"].insert(0, {
                "role": "system",
                "content": "You are a helpful assistant that responds in valid JSON format."
            })

        start_time = time.time()

        try:
            # Rate limiting
            self._wait_for_rate_limit()

            # Make API call
            logger.info(f"Making LLM request with model: {model.value}")
            response = self.client.chat.completions.create(**params)

            end_time = time.time()

            # Extract response content
            content = response.choices[0].message.content

            # Count tokens (approximate)
            tokens_used = len(content.split()) * 1.3

            self.request_count += 1

            return LLMResponse(
                content=content,
                model=model.value,
                tokens_used=int(tokens_used),
                response_time=end_time - start_time,
                success=True
            )

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return LLMResponse(
                content=self._get_mock_response(prompt, json_mode),
                model=model.value,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        try:
            if response.strip().startswith('{'):
                return json.loads(response)

            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)

            return None

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return None

    def extract_insights(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract insights from academic paper"""
        system_prompt = """You are an expert academic paper analyzer. 
        Extract key information and provide responses in valid JSON format only.
        Be concise but comprehensive."""

        prompt = f"""
        Analyze this academic paper and extract the following information:
        1. Key contributions (2-3 main points)
        2. Main findings (2-3 key results)
        3. Technical approach (brief description)
        4. Research gap addressed

        Paper text (first 2000 chars):
        {text[:2000]}

        Abstract (if available):
        {sections.get('abstract', 'Not found')}

        Respond with ONLY a JSON object in this exact format:
        {{
            "key_contributions": ["contribution1", "contribution2"],
            "main_findings": ["finding1", "finding2"],
            "technical_approach": "brief description",
            "research_gap": "gap addressed"
        }}
        """

        response = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            task_type="extraction",
            json_mode=True
        )

        parsed = self.extract_json_from_response(response)
        if parsed:
            return parsed

        return {
            "key_contributions": ["Extraction failed - raw response available"],
            "main_findings": ["Extraction failed - raw response available"],
            "technical_approach": "Extraction failed - raw response available",
            "research_gap": "Extraction failed - raw response available",
            "raw_response": response
        }

    def classify_topic(self, prompt: str, topics: List[str], model_type: ModelType = ModelType.LLAMA_70B) -> str:
        """Convenience method for topic classification"""
        system_prompt = "You are a research paper classifier. Respond with ONLY the topic name from the provided list."
        
        full_prompt = f"""{prompt}

Available topics: {', '.join(topics)}

Instructions: Choose the MOST relevant topic from the list above. Respond with ONLY the topic name, nothing else.

Topic:"""
        
        try:
            response = self.get_response(
                prompt=full_prompt,
                model_type=model_type
            )
            
            # Clean the response
            response = response.strip()
            
            # Validate the response is one of the available topics
            for topic in topics:
                if topic.lower() in response.lower():
                    return topic
            
            # If no exact match, return the first topic as fallback
            logger.warning(f"Topic classification returned invalid response: {response}")
            return topics[0] if topics else "Unknown"
            
        except Exception as e:
            logger.error(f"Topic classification failed: {e}")
            return topics[0] if topics else "Unknown"

    def classify_paper_topic(self, paper_data: Dict[str, Any], available_topics: List[str]) -> str:
        """Classify a research paper into one of the available topics"""
        
        # Create a comprehensive prompt from paper data
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        existing_topics = paper_data.get('topics', [])
        
        prompt = f"""
        Research Paper Classification:
        
        Title: {title}
        
        Abstract: {abstract}
        
        Current Topics: {', '.join(existing_topics) if existing_topics else 'None'}
        
        Please classify this research paper into the most appropriate topic from the available list.
        """
        
        return self.classify_topic(prompt, available_topics)
