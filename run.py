import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from collections import defaultdict
from agents.fetcher_agent import handle_user_input
from agents.extractor_agent import ExtractorAgent
from agents.classifier_agent import TopicClassifierAgent
from agents.topicsynthesizer import TopicSynthesizerAgent
from agents.media_citation_agent import MediaAndCitationAgent, TEMP_AUDIO_DIR
from core.llm_manager import LLMManager, ModelType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_pdf_summary(topic_syntheses, citations, output_dir="output"):
    """
    Generate a PDF summary from topic syntheses and citations
    Returns the PDF file path
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate PDF filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"research_summary_{timestamp}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Research Paper Analysis Summary", title_style))
        story.append(Spacer(1, 20))
        
        # Topic Syntheses
        if topic_syntheses:
            story.append(Paragraph("Topic Syntheses", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for synthesis in topic_syntheses:
                story.append(Paragraph(f"<b>Topic: {synthesis['topic']}</b>", styles['Heading3']))
                story.append(Paragraph(synthesis['synthesis'], styles['Normal']))
                story.append(Spacer(1, 12))
            
            story.append(PageBreak())
        
        # Citations
        if citations:
            story.append(Paragraph("Citations", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for i, citation in enumerate(citations, 1):
                story.append(Paragraph(f"{i}. {citation}", styles['Normal']))
                story.append(Spacer(1, 8))
        
        # Build PDF
        doc.build(story)
        return pdf_path
        
    except ImportError:
        logger.warning("reportlab not installed. Cannot generate PDF. Install with: pip install reportlab")
        return None
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None

def create_json_output(topic_syntheses, all_paper_metadata, pdf_path=None):
    """
    Create JSON output in the specified format
    """
    # Create comprehensive summary from all topic syntheses - include full content
    summary_parts = []
    
    if topic_syntheses:
        for synthesis in topic_syntheses:
            topic = synthesis['topic']
            content = synthesis['synthesis']
            # Include full synthesis content
            summary_parts.append(f"TOPIC: {topic}\n{content}")
    
    # If no syntheses, create summary from paper metadata
    if not summary_parts and all_paper_metadata:
        summary_parts = [paper.get('summary', '') for paper in all_paper_metadata]
    
    # Combine all summary parts with proper formatting
    combined_summary = "\n\n".join(summary_parts) if summary_parts else "Research paper analysis completed."
    
    # Extract unique links from paper metadata
    links = []
    for paper in all_paper_metadata:
        doi = paper.get('doi', '')
        if doi:
            if doi.startswith('10.'):
                links.append(f"https://doi.org/{doi}")
            elif doi.startswith('http'):
                links.append(doi)
    
    # Add arXiv links if available
    for paper in all_paper_metadata:
        title = paper.get('title', '').lower()
        if 'arxiv' in title or any('arxiv' in str(author).lower() for author in paper.get('authors', [])):
            # This is a placeholder - you'd need to extract actual arXiv IDs
            links.append("https://arxiv.org/abs/example")
    
    # Remove duplicates and limit to reasonable number
    links = list(set(links))[:10]
    
    # Create JSON structure
    json_output = {
        "summary": combined_summary,
        "links": links,
        "pdf_url": pdf_path if pdf_path else None
    }
    
    return json_output

def save_json_output(json_data, filename="research_output.json"):
    """
    Save JSON data to file
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON output saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving JSON output: {e}")
        return None

def main():
    """Main execution function"""
    try:
        # Configuration
        input_query = "YOLO"
        max_papers = 3
        user_defined_topics = [
            "Computer Vision", 
            "Healthcare", 
            "Robotics", 
            "Natural Language Processing", 
            "Autonomous Vehicles",
            "AI Safety",
            "Surveillance"
        ]

        logger.info(f"Starting paper fetching for query: '{input_query}'")
        papers = handle_user_input(input_query, max_papers=max_papers)

        if not papers:
            logger.error("❌ No papers found or downloaded.")
            print("\n🔍 Suggestions:")
            print("1. Try a more specific query")
            print("2. Use a specific DOI if you have one")
            print("3. Try different keywords")
            return

        # Initialize LLM, extractor, and classifier
        try:
            llm = LLMManager(api_key="f34f6f2101bcbbd56e99c79e889cb8289b335bee889570178d637cc926fcba00")
            extractor = ExtractorAgent(llm)
            classifier = TopicClassifierAgent(llm_manager=llm, model_type=ModelType.LLAMA_70B)
            synthesizer = TopicSynthesizerAgent(llm)
            media_agent = MediaAndCitationAgent(audio_language="en", slow=False)
        except Exception as e:
            logger.error(f"Failed to initialize LLM/Agents: {e}")
            print("❌ Initialization failed. Please check your API key and internet connection.")
            return

        # Store processed papers by topic
        papers_by_topic = defaultdict(list)
        all_paper_metadata = []

        # Process each paper
        successful_extractions = 0
        for i, paper in enumerate(papers, 1):
            print(f"\n{'='*60}")
            print(f"📄 Processing Paper {i}/{len(papers)}")
            print(f"Title: {paper.get('title', 'Unknown')}")
            print(f"Authors: {', '.join(paper.get('authors', []))}")
            print(f"DOI: {paper.get('doi', 'N/A')}")
            print(f"Year: {paper.get('year', 'N/A')}")
            print(f"Journal: {paper.get('journal', 'N/A')}")

            pdf_path = paper.get("pdf_path")
            if not pdf_path:
                print("⚠️ No PDF available - skipping extraction")
                continue

            print(f"📁 PDF Path: {pdf_path}")
            print(f"🔄 Starting extraction...")

            try:
                result = extractor.process_paper(paper)

                if result:
                    print(f"✅ Extraction successful!")
                    print(f"📊 Quality Score: {result.extraction_quality['confidence']:.2f}")
                    print(f"📄 Sections Found: {list(result.sections.keys())}")
                    print(f"📝 Text Length: {len(result.full_text)} characters")

                    insights = result.llm_enhanced
                    print(f"\n🧠 Key Insights:")
                    print(f"   Contributions: {insights.get('key_contributions', ['N/A'])}")
                    print(f"   Findings: {insights.get('main_findings', ['N/A'])}")
                    print(f"   Approach: {insights.get('technical_approach', 'N/A')}")

                    # Classification Step
                    paper["title"] = paper.get("title", "")
                    paper["abstract"] = result.sections.get("abstract", "")
                    paper["topics"] = user_defined_topics

                    classification = classifier.classify(paper)
                    predicted_topic = classification['predicted_topic']
                    confidence = classification['confidence']
                    
                    print(f"\n📌 Predicted Topic: {predicted_topic} (Confidence: {confidence})")

                    # Store paper with its classification and insights
                    paper_summary = {
                        "title": paper.get("title", "Unknown"),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year", "N/A"),
                        "journal": paper.get("journal", "N/A"),
                        "doi": paper.get("doi", "N/A"),
                        "abstract": result.sections.get("abstract", ""),
                        "summary": insights.get("summary", ""),
                        "key_contributions": insights.get("key_contributions", []),
                        "main_findings": insights.get("main_findings", []),
                        "technical_approach": insights.get("technical_approach", ""),
                        "predicted_topic": predicted_topic,
                        "confidence": confidence
                    }
                    
                    papers_by_topic[predicted_topic].append(paper_summary)
                    all_paper_metadata.append(paper_summary)
                    successful_extractions += 1
                else:
                    print("❌ Extraction failed")

            except Exception as e:
                logger.error(f"Error processing paper: {e}")
                print(f"❌ Error during extraction: {e}")

        print(f"\n{'='*60}")
        print(f"📊 Summary: {successful_extractions}/{len(papers)} papers processed successfully")

        if successful_extractions == 0:
            print("\n💡 Troubleshooting tips:")
            print("1. Check if PDFs are corrupted")
            print("2. Verify API key is correct")
            print("3. Try with a direct PDF URL or DOI")
            return

        # Topic Synthesis Step
        print(f"\n{'='*60}")
        print("🧩 Starting Topic Synthesis...")
        
        topic_syntheses = []
        for topic, topic_papers in papers_by_topic.items():
            if len(topic_papers) >= 1:
                print(f"\n🔄 Synthesizing topic: {topic} ({len(topic_papers)} papers)")
                
                synthesis_result = synthesizer.synthesize(topic, topic_papers)
                topic_syntheses.append(synthesis_result)
                
                print(f"✅ Synthesis complete for {topic}")
                print(f"📝 Preview: {synthesis_result['synthesis'][:200]}...")
            else:
                print(f"⚠️ Skipping {topic} - not enough papers ({len(topic_papers)})")

        # Generate and Save Text Files
        print(f"\n{'='*60}")
        print("📄 Generating Text Files...")
        
        # Generate citations for all papers
        citations = media_agent.generate_citations(all_paper_metadata)
        print(f"✅ Generated {len(citations)} citations")
        
        # Save citations to file
        try:
            with open("citations.txt", "w", encoding="utf-8") as f:
                f.write("RESEARCH PAPER CITATIONS\n")
                f.write("=" * 50 + "\n\n")
                for i, citation in enumerate(citations, 1):
                    f.write(f"{i}. {citation}\n\n")
            print("📄 Citations saved to 'citations.txt'")
        except Exception as e:
            print(f"❌ Error saving citations: {e}")
        
        # Save topic syntheses to file
        try:
            with open("topic_syntheses.txt", "w", encoding="utf-8") as f:
                f.write("TOPIC SYNTHESIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                for synthesis in topic_syntheses:
                    f.write(f"TOPIC: {synthesis['topic']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{synthesis['synthesis']}\n\n")
                    f.write("=" * 50 + "\n\n")
            print("📄 Topic syntheses saved to 'topic_syntheses.txt'")
        except Exception as e:
            print(f"❌ Error saving topic syntheses: {e}")

        # Generate PDF Summary
        print(f"\n{'='*60}")
        print("📄 Generating PDF Summary...")
        
        pdf_path = generate_pdf_summary(topic_syntheses, citations)
        if pdf_path:
            print(f"✅ PDF summary generated: {pdf_path}")
        else:
            print("⚠️ PDF generation failed or skipped")

        # Create JSON Output
        print(f"\n{'='*60}")
        print("📊 Creating JSON Output...")
        
        json_output = create_json_output(topic_syntheses, all_paper_metadata, pdf_path)
        
        # Save JSON output
        json_filename = save_json_output(json_output)
        if json_filename:
            print(f"✅ JSON output saved to: {json_filename}")
            print(f"📄 JSON Preview:")
            print(json.dumps(json_output, indent=2)[:500] + "...")
        else:
            print("❌ Failed to save JSON output")

        # Audio Generation (with proper error handling)
        print(f"\n{'='*60}")
        print("🎵 Starting Audio Generation...")
        
        audio_files_generated = []
        
        try:
            # Generate audio for topic_syntheses.txt (single consolidated file)
            print("\n🔄 Generating audio for topic_syntheses.txt...")
            topic_syntheses_audio = media_agent.generate_audio_for_syntheses_file()
            
            if topic_syntheses_audio:
                audio_files_generated.append(topic_syntheses_audio)
                print(f"✅ Generated consolidated audio: {topic_syntheses_audio}")
            else:
                print("⚠️ Failed to generate audio for topic_syntheses.txt")
        except Exception as e:
            print(f"❌ Error generating audio for syntheses: {e}")
        
        try:
            # Generate audio for citations
            print("🔄 Generating audio for citations...")
            citations_audio = media_agent.generate_audio_for_citations(citations)
            if citations_audio:
                audio_files_generated.append(citations_audio)
                print("✅ Generated citations audio file")
            else:
                print("⚠️ No audio file generated for citations")
        except Exception as e:
            print(f"❌ Error generating audio for citations: {e}")
        
        print(f"\n📊 Audio Generation Summary:")
        print(f"🎵 Audio files generated: {len(audio_files_generated)}")
        
        if audio_files_generated:
            print(f"📁 Audio files saved in: {TEMP_AUDIO_DIR}/")
            for audio_path in audio_files_generated:
                if audio_path and os.path.exists(audio_path):
                    print(f"   🎵 {os.path.basename(audio_path)}")
        else:
            print("⚠️ No audio files were successfully generated")

        # Final Summary
        print(f"\n{'='*60}")
        print("🎉 PIPELINE COMPLETE!")
        print(f"📊 Papers processed: {successful_extractions}")
        print(f"🧩 Topics synthesized: {len(topic_syntheses)}")
        print(f"📄 Citations generated: {len(citations)}")
        print(f"🎵 Audio files generated: {len(audio_files_generated)}")
        print(f"📁 Output files:")
        print(f"   📄 citations.txt")
        print(f"   📄 topic_syntheses.txt")
        print(f"   📊 {json_filename}")
        if pdf_path:
            print(f"   📄 {pdf_path}")
        if audio_files_generated:
            print(f"   🎵 {TEMP_AUDIO_DIR}/")
            
        # Print final JSON output
        print(f"\n{'='*60}")
        print("📊 FINAL JSON OUTPUT:")
        print(json.dumps(json_output, indent=2))

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()