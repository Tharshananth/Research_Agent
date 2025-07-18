
import os
from typing import List, Dict
from gtts import gTTS

TEMP_AUDIO_DIR = "generated_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

class MediaAndCitationAgent:
    def __init__(self, audio_language="en", slow=False):
        """
        Initialize the MediaAndCitationAgent with gTTS
        
        Args:
            audio_language (str): Language code for TTS ("en", "hi", etc.)
            slow (bool): Whether to speak slowly
        """
        self.audio_language = audio_language
        self.slow = slow

    def generate_citations(self, paper_metadata: List[Dict]) -> List[str]:
        """Generate APA-style citations for each paper."""
        citations = []
        for paper in paper_metadata:
            try:
                # Handle multiple authors
                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    if len(authors) == 1:
                        author = authors[0]
                    elif len(authors) == 2:
                        author = f"{authors[0]} & {authors[1]}"
                    elif len(authors) > 2:
                        author = f"{authors[0]} et al."
                    else:
                        author = "Unknown Author"
                else:
                    author = str(authors) if authors else "Unknown Author"

                year = paper.get("year", "n.d.")
                title = paper.get("title", "Untitled")
                journal = paper.get("journal", "Unknown Journal")
                doi = paper.get("doi", "")

                if doi:
                    citation = f"{author} ({year}). {title}. *{journal}*. https://doi.org/{doi}"
                else:
                    citation = f"{author} ({year}). {title}. *{journal}*."
                
                citations.append(citation)

            except Exception as e:
                print(f"⚠️ Error generating citation: {e}")
                citations.append(f"Error generating citation for: {paper.get('title', 'Unknown')}")
        
        return citations

    def generate_audio_from_text(self, text: str, output_filename: str) -> str:
        """
        Generate audio directly from text string using gTTS.
        
        Args:
            text (str): Text to convert to speech
            output_filename (str): Name of the output audio file
            
        Returns:
            str: Path to the generated audio file, or None if failed
        """
        try:
            # Validate text input
            if not text or not text.strip():
                print("❌ Error: Empty text provided")
                return None
            
            # Limit text length (gTTS has character limits)
            if len(text) > 5000:
                print(f"⚠️ Text too long ({len(text)} chars), truncating to 5000 characters")
                text = text[:5000] + "..."
            
            print(f"🎵 Converting {len(text)} characters to speech...")
            
            # Convert text to speech using gTTS
            tts = gTTS(text=text, lang=self.audio_language, slow=self.slow)
            
            # Save audio file
            audio_path = os.path.join(TEMP_AUDIO_DIR, output_filename)
            tts.save(audio_path)
            
            print(f"✅ Audio saved to: {audio_path}")
            return audio_path
            
        except Exception as e:
            print(f"❌ Error converting text to audio: {e}")
            return None

    def generate_audio_from_file(self, text_file_path: str, output_filename: str) -> str:
        """
        Generate audio from a text file.
        
        Args:
            text_file_path (str): Path to the text file
            output_filename (str): Name of the output audio file
            
        Returns:
            str: Path to the generated audio file, or None if failed
        """
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return self.generate_audio_from_text(text, output_filename)
            
        except Exception as e:
            print(f"❌ Error reading text file or generating audio: {e}")
            return None

    def generate_audio_for_syntheses_file(self, input_file="topic_syntheses.txt", output_file="topic_syntheses.mp3"):
        """
        Generate audio from the topic_syntheses.txt file.
        
        Args:
            input_file (str): Path to the text file (default: "topic_syntheses.txt")
            output_file (str): Output audio filename (default: "topic_syntheses.mp3")
        
        Returns:
            str: Path to the generated audio file, or None if failed
        """
        try:
            # Read the entire file
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Skip if empty
            if not text.strip():
                print(f"⚠️ File {input_file} is empty!")
                return None
            
            print(f"🎵 Converting {len(text)} characters from {input_file} to speech...")
            
            # Generate audio
            return self.generate_audio_from_text(text, output_file)
        
        except FileNotFoundError:
            print(f"❌ File {input_file} not found!")
            return None
        except Exception as e:
            print(f"❌ Error generating audio from {input_file}: {e}")
            return None

    def generate_audio_for_citations(self, citations: List[str] = None) -> str:
        """
        Generate audio for citations.
        
        Args:
            citations (List[str]): List of citation strings. If None, reads from citations.txt
            
        Returns:
            str: Path to the generated audio file, or None if failed
        """
        try:
            # If no citations provided, try to read from citations.txt
            if citations is None:
                try:
                    with open("citations.txt", 'r', encoding='utf-8') as f:
                        citations_text = f.read()
                    
                    if not citations_text.strip():
                        print("⚠️ citations.txt is empty!")
                        return None
                    
                    # Add introduction
                    citations_text = "Research Paper Citations. " + citations_text
                    
                except FileNotFoundError:
                    print("❌ citations.txt not found!")
                    return None
            else:
                # Use provided citations list
                if not citations:
                    print("⚠️ No citations provided")
                    return None
                
                # Combine all citations into one text
                citations_text = "Research Paper Citations. " + " ".join(citations)
            
            # Limit text length
            if len(citations_text) > 5000:
                citations_text = citations_text[:5000] + "... and more citations available in the text file."
            
            return self.generate_audio_from_text(citations_text, "citations.mp3")
            
        except Exception as e:
            print(f"❌ Error generating audio for citations: {e}")
            return None

# Usage example function
def test_tts():
    """Test function to demonstrate TTS usage"""
    # Initialize the agent
    agent = MediaAndCitationAgent(
        audio_language="en",  # Use "hi" for Hindi
        slow=False
    )
    
    # Test text
    test_text = "Hello, this is a test of the Google text-to-speech system."
    
    # Generate audio
    audio_path = agent.generate_audio_from_text(test_text, "test_audio.mp3")
    
    if audio_path:
        print(f"✅ Test audio generated: {audio_path}")
    else:
        print("❌ Failed to generate test audio")

# Main execution example
if __name__ == "__main__":
    # Initialize the agent
    agent = MediaAndCitationAgent(audio_language="en", slow=False)
    
    # Test the audio generation
    print("🎵 Testing audio generation...")
    
    # Generate audio for topic syntheses
    print("\n🔄 Generating audio for topic_syntheses.txt...")
    syntheses_audio = agent.generate_audio_for_syntheses_file()
    if syntheses_audio:
        print(f"✅ Topic syntheses audio generated: {syntheses_audio}")
    else:
        print("❌ Failed to generate topic syntheses audio")
    
    # Generate audio for citations
    print("\n🔄 Generating audio for citations...")
    citations_audio = agent.generate_audio_for_citations()
    if citations_audio:
        print(f"✅ Citations audio generated: {citations_audio}")
    else:
        print("❌ Failed to generate citations audio")
    
    print("\n🎉 Audio generation process complete!")