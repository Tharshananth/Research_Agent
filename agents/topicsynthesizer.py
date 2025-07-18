

import logging
from typing import List, Dict, Any
from core.llm_manager import LLMManager, ModelType

class TopicSynthesizerAgent:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    def synthesize(self, topic: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not summaries:
            return {
                "topic": topic,
                "synthesis": "No summaries available to synthesize."
            }

        # Format summaries with better structure
        summaries_text = "\n\n".join([
            f"Paper {i+1}:\n"
            f"Title: {paper.get('title', 'Unknown')}\n"
            f"Authors: {', '.join(paper.get('authors', []))}\n"
            f"Year: {paper.get('year', 'N/A')}\n"
            f"Key Contributions: {', '.join(paper.get('key_contributions', []))}\n"
            f"Main Findings: {', '.join(paper.get('main_findings', []))}\n"
            f"Technical Approach: {paper.get('technical_approach', 'N/A')}\n"
            f"Abstract: {paper.get('abstract', 'No abstract available.')}"
            for i, paper in enumerate(summaries)
        ])

        prompt = (
            f"You are an expert research analyst specializing in academic paper synthesis.\n\n"
            f"TOPIC: {topic}\n\n"
            f"Below are research papers from this topic:\n\n"
            f"{summaries_text}\n\n"
            f"TASK: Write a comprehensive synthesis that:\n"
            f"1. Identifies key themes and trends across these papers\n"
            f"2. Compares and contrasts different approaches and findings\n"
            f"3. Highlights significant contributions and innovations\n"
            f"4. Discusses methodological approaches used\n"
            f"5. Summarizes the current state of research in this area\n\n"
            f"Keep the synthesis concise but comprehensive (300-500 words)."
        )

        try:
            logging.info(f"🔄 Synthesizing topic: {topic} with {len(summaries)} papers.")

            # ✅ MAIN FIX: Try the most common LLMManager patterns
            response = None
            
            # Method 1: Try get_model() pattern (like your classifier uses)
            if hasattr(self.llm_manager, 'get_model'):
                try:
                    model = self.llm_manager.get_model(ModelType.LLAMA_70B)
                    response = model.generate_response(prompt)
                    logging.info("✅ Used get_model() -> generate_response() pattern")
                except Exception as e:
                    logging.warning(f"get_model() pattern failed: {e}")
            
            # Method 2: Try direct model access
            if not response and hasattr(self.llm_manager, 'model'):
                try:
                    response = self.llm_manager.model.generate_response(prompt)
                    logging.info("✅ Used direct model access pattern")
                except Exception as e:
                    logging.warning(f"Direct model access failed: {e}")
            
            # Method 3: Try generate_response directly
            if not response and hasattr(self.llm_manager, 'generate_response'):
                try:
                    response = self.llm_manager.generate_response(prompt)
                    logging.info("✅ Used direct generate_response() pattern")
                except Exception as e:
                    logging.warning(f"Direct generate_response failed: {e}")
            
            # Method 4: Try query pattern
            if not response and hasattr(self.llm_manager, 'query'):
                try:
                    response = self.llm_manager.query(prompt)
                    logging.info("✅ Used query() pattern")
                except Exception as e:
                    logging.warning(f"Query pattern failed: {e}")
            
            # Method 5: Try get_response pattern
            if not response and hasattr(self.llm_manager, 'get_response'):
                try:
                    response = self.llm_manager.get_response(prompt)
                    logging.info("✅ Used get_response() pattern")
                except Exception as e:
                    logging.warning(f"get_response pattern failed: {e}")
            
            # Method 6: Try call pattern
            if not response and hasattr(self.llm_manager, 'call'):
                try:
                    response = self.llm_manager.call(prompt)
                    logging.info("✅ Used call() pattern")
                except Exception as e:
                    logging.warning(f"Call pattern failed: {e}")
            
            # Method 7: Try invoke pattern
            if not response and hasattr(self.llm_manager, 'invoke'):
                try:
                    response = self.llm_manager.invoke(prompt)
                    logging.info("✅ Used invoke() pattern")
                except Exception as e:
                    logging.warning(f"Invoke pattern failed: {e}")
            
            # Method 8: Try __call__ pattern
            if not response and hasattr(self.llm_manager, '__call__'):
                try:
                    response = self.llm_manager(prompt)
                    logging.info("✅ Used __call__ pattern")
                except Exception as e:
                    logging.warning(f"__call__ pattern failed: {e}")

            # If still no response, show debugging info
            if not response:
                methods = [method for method in dir(self.llm_manager) if not method.startswith('_')]
                logging.error(f"❌ All methods failed. Available methods: {methods}")
                
                # Try to find any method that might work
                for method_name in methods:
                    if any(keyword in method_name.lower() for keyword in ['generate', 'response', 'query', 'call', 'invoke']):
                        try:
                            method = getattr(self.llm_manager, method_name)
                            if callable(method):
                                response = method(prompt)
                                logging.info(f"✅ Found working method: {method_name}")
                                break
                        except Exception as e:
                            logging.warning(f"Method {method_name} failed: {e}")
                            continue

            return {
                "topic": topic,
                "synthesis": response.strip() if response else "No response generated - please check LLM configuration."
            }

        except Exception as e:
            logging.error(f"❌ Error during topic synthesis: {e}")
            logging.error(f"LLM Manager type: {type(self.llm_manager)}")
            logging.error(f"LLM Manager methods: {[m for m in dir(self.llm_manager) if not m.startswith('_')]}")
            
            return {
                "topic": topic,
                "synthesis": f"Synthesis failed due to an internal error: {str(e)}"
            }