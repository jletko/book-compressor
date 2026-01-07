import os
import argparse
import math
import time
import concurrent.futures
import io
import base64
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import pypdf
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from PIL import Image

# Constants
MAX_OUTPUT_TOKENS = 8192  
SAFE_OUTPUT_WORDS = 2000  # Maximum target output per chunk (prevents exceeding model limits)
MAX_INPUT_WORDS_PER_CHUNK = 6000  # Maximum input words per chunk (prevents hallucinations)
MINIMUM_OUTPUT_WORDS_PER_CHUNK = 20  # Minimum output words per chunk (ensures meaningful content)
WORDS_PER_MINUTE = 150 # Standard TTS/Audiobook speed
# OpenRouter model IDs
MODEL_NAME = 'google/gemini-3-flash-preview'  # Main compression model
HELPER_MODEL_NAME = 'google/gemini-3-flash-preview'  # Used for book analysis and title generation
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Safety Settings: Block None to allow processing of all content (e.g. mature themes in books)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

class PageContent:
    def __init__(self, text: str, images: List[str] | None = None, is_chapter_start: bool = False):
        self.text = text
        self.images = images if images else [] # List of base64 strings
        self.is_chapter_start = is_chapter_start

class TokenLimitExceededError(Exception):
    pass

class EmptyContentError(Exception):
    pass

def setup_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("API Key not found. Please set the OPENROUTER_API_KEY environment variable.")
    
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

def extract_images_from_page(page: pypdf.PageObject) -> List[str]:
    """Extracts images from a PDF page and returns them as base64 strings."""
    images: List[str] = []
    try:
        resources = page.get('/Resources')
        if resources and '/XObject' in resources:  # type: ignore
            xObject = resources['/XObject'].get_object()  # type: ignore
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':  # type: ignore
                    try:
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        if size[0] < 100 or size[1] < 100: continue
                        
                        data = xObject[obj].get_data()
                        image = Image.open(io.BytesIO(data))
                        if image.mode != 'RGB': image = image.convert('RGB')
                            
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG", quality=85)
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        images.append(img_str)
                    except Exception:
                        continue 
    except Exception:
        pass 
    return images

def get_pdf_chapter_starts(reader: pypdf.PdfReader) -> set:
    """Extracts page numbers that correspond to chapter starts from PDF outline."""
    chapter_starts = set()
    
    def _recurse_outline(outline):
        for item in outline:
            if isinstance(item, list):
                _recurse_outline(item)
            elif hasattr(item, 'page') and item.page is not None:
                # pypdf returns a PageObject or an index, usually we need to find index
                try:
                    page_index = reader.get_page_number(item.page)
                    if page_index is not None:
                        chapter_starts.add(page_index)
                except Exception:
                    pass
                    
    try:
        if reader.outline:
            _recurse_outline(reader.outline)
    except Exception:
        pass # Outline might be missing or complex
        
    return chapter_starts

def read_pdf(filepath: str, extract_images: bool = False) -> List[PageContent]:
    pages_content = []
    try:
        reader = pypdf.PdfReader(filepath)
        chapter_starts = get_pdf_chapter_starts(reader)
        if chapter_starts:
            print(f"  > Found {len(chapter_starts)} chapter markers in PDF metadata.")
        else:
            print(f"  > No chapter markers found in PDF (will use heuristic/math split).")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            images = []
            if extract_images:
                images = extract_images_from_page(page)
            
            is_start = i in chapter_starts
            pages_content.append(PageContent(text, images, is_chapter_start=is_start))
            
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")
    return pages_content

def read_epub(filepath: str, extract_images: bool = False) -> List[PageContent]:
    pages_content = []
    try:
        book = epub.read_epub(filepath)
        
        # Build image lookup map if extracting images
        image_map = {}
        if extract_images:
            for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
                # Store by filename and full path for flexible matching
                filename = os.path.basename(item.get_name())
                image_map[filename] = item
                image_map[item.get_name()] = item
                # Also store normalized path (without leading ../ or ./)
                normalized = item.get_name().lstrip('./').lstrip('../')
                image_map[normalized] = item
        
        items = list(book.get_items())
        doc_count = 0
        total_images = 0
        
        for item in items:
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                doc_count += 1
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                
                images = []
                if extract_images:
                    # Find all img tags
                    for img_tag in soup.find_all('img'):
                        src = img_tag.get('src', '')
                        if not src or not isinstance(src, str):
                            continue
                        
                        # Try to match image source to image items
                        img_item = None
                        src_filename = os.path.basename(src)
                        src_normalized = src.lstrip('./').lstrip('../')
                        
                        # Try different matching strategies
                        for key in [src, src_filename, src_normalized]:
                            if key in image_map:
                                img_item = image_map[key]
                                break
                        
                        if img_item:
                            try:
                                img_data = img_item.get_content()
                                image = Image.open(io.BytesIO(img_data))
                                
                                # Filter small images (icons, bullets, etc.)
                                if image.width < 100 or image.height < 100:
                                    continue
                                
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                
                                buffered = io.BytesIO()
                                image.save(buffered, format="JPEG", quality=85)
                                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                images.append(img_str)
                                total_images += 1
                            except Exception:
                                continue
                
                pages_content.append(PageContent(text, images, is_chapter_start=True))
        
        print(f"  > Found {doc_count} internal documents (potential chapters) in EPUB.")
        if extract_images:
            print(f"  > Extracted {total_images} images from EPUB.")
    except Exception as e:
        raise ValueError(f"Failed to read EPUB: {e}")
    return pages_content

def read_file(filepath: str, extract_images: bool = False) -> List[PageContent]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        print("  > Detected: PDF format")
        return read_pdf(filepath, extract_images)
    elif ext == '.epub':
        print("  > Detected: EPUB format")
        return read_epub(filepath, extract_images)
    else:
        raise ValueError("Unsupported file format. Only .pdf and .epub are supported.")

def write_file(filepath: str, content: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def count_words(text: str) -> int:
    return len(text.split())

def count_total_words(pages: List[PageContent]) -> int:
    return sum(count_words(p.text) for p in pages)

class Chunk:
    def __init__(self, text: str, images: List[str]):
        self.text = text
        self.images = images

def split_chunk_at_boundary(chunk: Chunk, max_words: int) -> List[Chunk]:
    """Splits a chunk into smaller pieces at paragraph/sentence boundaries."""
    text = chunk.text
    total_words = count_words(text)
    
    # Calculate how many sub-chunks we need
    num_splits = math.ceil(total_words / max_words)
    target_words_per_split = total_words // num_splits
    
    # Try paragraph boundaries first (double newline)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        # Distribute paragraphs into sub-chunks
        sub_chunks = []
        current_text = []
        current_words = 0
        
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue
                
            para_words = count_words(para)
            
            # If single paragraph exceeds max_words, split it further at sentence boundaries
            if para_words > max_words:
                # Save current accumulation first
                if current_text:
                    sub_chunks.append(Chunk('\n\n'.join(current_text), []))
                    current_text = []
                    current_words = 0
                
                # Split oversized paragraph at sentences
                import re
                sentences = re.split(r'([.!?]\s+)', para)
                sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
                
                sent_group = []
                sent_words = 0
                for sent in sentences:
                    if not sent.strip():
                        continue
                    sw = count_words(sent)
                    if sent_group and sent_words + sw > max_words:
                        sub_chunks.append(Chunk(''.join(sent_group), []))
                        sent_group = []
                        sent_words = 0
                    sent_group.append(sent)
                    sent_words += sw
                if sent_group:
                    sub_chunks.append(Chunk(''.join(sent_group), []))
                continue
            
            # Lenient limit: If adding this paragraph exceeds target, save current chunk
            if current_text and current_words + para_words > target_words_per_split * 1.2:
                sub_chunks.append(Chunk('\n\n'.join(current_text), []))
                current_text = []
                current_words = 0
            
            current_text.append(para)
            current_words += para_words
        
        # Add remaining
        if current_text:
            sub_chunks.append(Chunk('\n\n'.join(current_text), []))
        
        # Filter out any empty chunks
        sub_chunks = [sc for sc in sub_chunks if count_words(sc.text) > 0]
        
        # Distribute images evenly
        if chunk.images and len(sub_chunks) > 0:
            images_per_chunk = len(chunk.images) // len(sub_chunks)
            for i, sub_chunk in enumerate(sub_chunks):
                start_idx = i * images_per_chunk
                end_idx = start_idx + images_per_chunk if i < len(sub_chunks) - 1 else len(chunk.images)
                sub_chunk.images = chunk.images[start_idx:end_idx]
        
        return sub_chunks if sub_chunks else [chunk]
    
    # Fallback: sentence boundaries
    import re
    sentences = re.split(r'([.!?]\s+)', text)
    # Rejoin split markers with sentences
    sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
    if len(sentences) > 1:
        sub_chunks = []
        current_text = []
        current_words = 0
        
        for sent in sentences:
            # Skip empty sentences
            if not sent.strip():
                continue
                
            sent_words = count_words(sent)
            
            # Lenient limit
            if current_text and current_words + sent_words > target_words_per_split * 1.2:
                sub_chunks.append(Chunk(''.join(current_text), []))
                current_text = []
                current_words = 0
            
            current_text.append(sent)
            current_words += sent_words
        
        if current_text:
            sub_chunks.append(Chunk(''.join(current_text), []))
        
        # Filter out any empty chunks
        sub_chunks = [sc for sc in sub_chunks if count_words(sc.text) > 0]
        
        # Distribute images
        if chunk.images and len(sub_chunks) > 0:
            images_per_chunk = len(chunk.images) // len(sub_chunks)
            for i, sub_chunk in enumerate(sub_chunks):
                start_idx = i * images_per_chunk
                end_idx = start_idx + images_per_chunk if i < len(sub_chunks) - 1 else len(chunk.images)
                sub_chunk.images = chunk.images[start_idx:end_idx]
        
        return sub_chunks if sub_chunks else [chunk]
    
    # Last resort: just split at word boundary near midpoint
    words = text.split()
    mid = len(words) // 2
    first_half = ' '.join(words[:mid])
    second_half = ' '.join(words[mid:])
    
    return [
        Chunk(first_half, chunk.images[:len(chunk.images)//2]),
        Chunk(second_half, chunk.images[len(chunk.images)//2:])
    ]

def split_into_chunks(pages: List[PageContent], num_chunks: int) -> List[Chunk]:
    """Splits pages into chunks using Semantic Logic (Chapters) + Word Limits."""
    total_words = count_total_words(pages)
    if total_words == 0: return []
    
    target_per_chunk = math.ceil(total_words / num_chunks)
    print(f"  > Semantic Chunking: Target ~{target_per_chunk} words/chunk")
    
    # Work on a copy to avoid modifying the original list
    pages = list(pages)
    
    # Pre-processing: Merge tiny leading pages (title pages, dedications) with the next substantial page
    min_words_threshold = max(50, int(target_per_chunk * 0.1))  # At least 50 words or 10% of target
    while len(pages) >= 2 and count_words(pages[0].text) < min_words_threshold:
        first_words = count_words(pages[0].text)
        print(f"  > Merging tiny leading page ({first_words} words) with next page.")
        # Merge first into second
        pages[1].text = pages[0].text + "\n" + pages[1].text
        pages[1].images = pages[0].images + pages[1].images
        # Keep is_chapter_start from second page (the real chapter)
        pages.pop(0)
    
    chunks = []
    current_text = []
    current_images = []
    current_words = 0
    
    for page in pages:
        words_in_page = count_words(page.text)
        
        # LOGIC:
        # 1. Soft Limit: Try to break at Chapter Start (page.is_chapter_start)
        #    BUT only if we have accumulated "enough" content (e.g., > 60% of target).
        #    This prevents creating tiny chunks for short chapters.
        # 2. Hard Limit: If we are WAY over target (e.g., > 120%), force split even mid-chapter.
        
        should_split = False
        
        # Case A: Chapter Boundary + Enough Content
        if page.is_chapter_start and current_words > (target_per_chunk * 0.6):
            should_split = True
            
        # Case B: Hard Limit Reached (Chapter is too long)
        elif current_words + words_in_page > (target_per_chunk * 1.25):
             should_split = True
             
        if should_split and current_words > 0:
            chunks.append(Chunk("\n".join(current_text), current_images))
            current_text = []
            current_images = []
            current_words = 0
            
        current_text.append(page.text)
        current_images.extend(page.images)
        current_words += words_in_page
        
    # Add final chunk logic
    if current_text:
        # Check if remainder is very small (e.g. < 25% of target) AND we have a previous chunk
        if chunks and current_words < (target_per_chunk * 0.25):
             print(f"  > Merging tiny remainder ({current_words} words) into previous chunk.")
             chunks[-1].text += "\n" + "\n".join(current_text)
             # Note: Image merging requires careful handling if extending lists
             if chunks[-1].images is None: chunks[-1].images = []
             chunks[-1].images.extend(current_images)
        else:
             chunks.append(Chunk("\n".join(current_text), current_images))
        
    return chunks

def load_prompt_template(prompt_path: str = "prompt.md") -> str:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file '{prompt_path}' not found. Please ensure it exists in the same directory.")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_chunk_title(text: str, chunk_number: int, target_language: str, client: OpenAI, model: str, style_instruction: str = "") -> str:
    """Generates a creative chapter title for a text chunk."""
    try:
        # Load prompt from file
        prompt_template = load_prompt_template(os.path.join("prompts", "prompt_generate_chunk_title.md"))
        
        # Use FULL text for maximum context (Gemini 1.5 Flash supports 1M+ tokens)
        text_sample = text 
            
        prompt = prompt_template.format(
            target_language=target_language, 
            text_chunk=text_sample, 
            chunk_number=chunk_number,
            style_specific_instruction=style_instruction
        )

        response = client.chat.completions.create(
            model=HELPER_MODEL_NAME,
            messages=[{
                "role": "user", 
                "content": prompt
            }],
            extra_body={"safetySettings": SAFETY_SETTINGS}
        )
        content = response.choices[0].message.content
        return content.strip() if content else f"CHAPTER {chunk_number}"
    except Exception:
        return f"CHAPTER {chunk_number}"

def compress_chunk_with_template(chunk: Chunk, chunk_number: int, target_word_count: int, client: OpenAI, model_name: str, language: str, prompt_template: str, temperature: float | None = None, title_instruction: str = "", focus_instruction: str = "", max_attempts: int = 2, output_format: str = "tts") -> str:
    """Compresses a single chunk using Best-of-N sampling: multiple attempts, keep best."""
    
    # Configuration
    WORD_COUNT_TOLERANCE = 0.15  # 15% - exit early if within this
    
    # Build prompt
    prompt_text = prompt_template.format(
        target_word_count=target_word_count,
        language=language,
        text_chunk=chunk.text,
        focus_instruction=focus_instruction
    )
    
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt_text})
    
    # Add Images (max 5)
    selected_images = chunk.images[:5] 
    if selected_images:
        print(f"  > Attaching {len(selected_images)} images to this chunk...")
    for img_b64 in selected_images:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    try:
        kwargs = {
            "model": model_name,
            "messages": messages,
            "extra_body": {"safetySettings": SAFETY_SETTINGS}
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Best-of-N sampling with adaptive target compensation
        best_content = None
        best_deviation = float('inf')
        adjusted_target = target_word_count  # Start with original target
        last_word_count = None
        
        for attempt in range(max_attempts):
            # Early exit if already within tolerance
            if best_deviation <= WORD_COUNT_TOLERANCE:
                break
            
            # Rebuild prompt with adjusted target for this attempt
            prompt_text = prompt_template.format(
                target_word_count=adjusted_target,
                language=language,
                text_chunk=chunk.text,
                focus_instruction=focus_instruction
            )
            messages[0]["content"][0] = {"type": "text", "text": prompt_text}
            kwargs["messages"] = messages
            
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            if not content or not content.strip():
                print(f"  > Chunk {chunk_number} attempt {attempt + 1}: Empty response, skipping.")
                continue
            
            word_count = count_words(content)
            # Deviation is always measured against ORIGINAL target
            deviation = abs(target_word_count - word_count) / target_word_count if target_word_count > 0 else 0
            
            print(f"  > Chunk {chunk_number} attempt {attempt + 1}/{max_attempts}: {word_count} words (asked: {adjusted_target}, target: {target_word_count}, deviation: {deviation:.1%})")
            
            if deviation < best_deviation:
                best_content = content
                best_deviation = deviation
            
            # Ratio-based compensation with dampening for next attempt
            # If model produces 80% of requested, scale up by inverse ratio
            if word_count > 0:
                ratio = word_count / adjusted_target  # How much model produced vs what we asked
                # Dampened correction: only apply 50% of the needed adjustment
                correction_factor = 1 / ratio
                dampened_factor = 1 + (correction_factor - 1) * 0.5  # 50% dampening
                adjusted_target = int(target_word_count * dampened_factor)
                # Clamp to reasonable bounds
                adjusted_target = max(adjusted_target, int(target_word_count * 0.5))
                adjusted_target = min(adjusted_target, int(target_word_count * 1.5))
        
        if best_content is None:
            error_msg = f"FATAL: All {max_attempts} attempts returned empty content for chunk {chunk_number}."
            raise EmptyContentError(error_msg)
        
        compressed_content = best_content
        
        # Step 2: Generate Title from Compressed Output
        print(f"  > Generating Title for Chunk {chunk_number}...")
        chunk_title = generate_chunk_title(compressed_content, chunk_number, language, client, model_name, style_instruction=title_instruction)
        print(f"  > Title {chunk_number}: {chunk_title}")
        
        # Step 3: Format Output with Generated Title
        # Ensure newlines for spacing
        if output_format == "reading":
            # Add Markdown Header for 'reading' format
            chunk_title = f"# {chunk_title}"
            
        final_output = f"{chunk_title}\n\n{compressed_content}"
        return final_output

    except TokenLimitExceededError:
        raise # Allow this specific error to bubble up to the main executor loop
    except Exception as e:
        print(f"Error compressing chunk: {e}")
        return f"[ERROR COMPRESSING CHUNK: {e}]"

def analyze_book_type(pages: List[PageContent], client: OpenAI, model_name: str) -> str:
    """Analyzes the text sample to determine if it's FICTION or NONFICTION."""
    
    total_pages = len(pages)
    text_sample = ""

    if total_pages < 10:
        # For very short books, just use everything
        text_sample = "\n".join([p.text for p in pages])
    else:
        # Distributed Sampling Strategy
        # Aim for ~20% coverage, split into 3 segments
        # Minimum 5 pages total sample
        sample_page_count = max(int(total_pages * 0.20), 5)
        pages_per_segment = max(sample_page_count // 3, 1) # At least 1 page per segment
        
        # Define start percentages: Start (10%), Middle (45%), End (80%)
        # This avoids front matter and end matter/appendices
        start_indices = [
            int(total_pages * 0.10),
            int(total_pages * 0.45),
            int(total_pages * 0.80)
        ]
        
        segments = []
        print(f"  > Sampling book for analysis: {sample_page_count} pages total from 3 locations...")
        
        for i, start_idx in enumerate(start_indices):
            # Ensure we don't go out of bounds
            if start_idx >= total_pages:
                continue
                
            end_idx = min(start_idx + pages_per_segment, total_pages)
            
            # Extract text
            segment_text = "\n".join([p.text for p in pages[start_idx:end_idx]])
            segments.append(f"--- SAMPLE SEGMENT {i+1} ---")
            segments.append(segment_text)
            
        text_sample = "\n\n".join(segments)

    # Truncate if huge to fit context (e.g. 50k chars is plenty for classification)
    text_sample = text_sample[:50000]
    
    try:
        prompt_template = load_prompt_template(os.path.join("prompts", "prompt_classifier.md"))
        prompt = prompt_template.format(text_sample=text_sample)
        
        response = client.chat.completions.create(
            model=HELPER_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "safetySettings": SAFETY_SETTINGS
            }
        )
        content = response.choices[0].message.content
        result = content.strip().upper() if content else "FICTION"
        if "NONFICTION" in result: return "NONFICTION"
        return "FICTION" 
    except Exception as e:
        print(f"Error analyzing book type: {e}. Defaulting to FICTION.")
        return "FICTION"

def process_file(filename: str, args, client: OpenAI):
    input_path = os.path.join("Input", filename)
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")
    
    # Logic optimization: Determine if we need images BEFORE reading if possible
    # to avoid double-reading for forced styles.
    
    enable_vision = False
    output_format = args.format
    
    # Initial read - no images yet, we need to detect content type first
    extract_images_initial = False
    
    try:
        pages = read_file(input_path, extract_images=extract_images_initial)
    except Exception as e:
        print(f"ERROR: Failed to read {filename} - {e}")
        return

    # Smart Mode / Style Detection
    # Logic: 
    # 1. ALWAYS detect book type (Fiction/NonFiction) to determine Title Generation Strategy
    # 2. Use detected_book_type to determine if images are needed
    
    print("\n" + "="*50)
    print("ANALYZING BOOK")
    print("="*50)
    detected_book_type = analyze_book_type(pages, client, args.model)
    print(f"  > Content Type: {detected_book_type}")
    print(f"  > Output Format: {output_format.upper()}")

    # Image extraction logic:
    # - Non-fiction: ALWAYS extract images (diagrams, charts are important)
    # - Fiction: NEVER extract images (no relevant visuals)
    
    supports_images = filename.lower().endswith('.pdf') or filename.lower().endswith('.epub')
    if detected_book_type == "NONFICTION" and supports_images:
         print(f"  > Re-reading {os.path.splitext(filename)[1].upper()} with image extraction...")
         try:
             pages = read_file(input_path, extract_images=True)
             enable_vision = True
         except Exception as e:
             print(f"Warning: Re-read failed: {e}")

    # Apply limit AFTER potentially re-reading
    full_word_count = count_total_words(pages)
    
    if args.process_limit < 100.0:
        if not (0.1 <= args.process_limit <= 100.0):
            print("ERROR: process_limit must be between 0.1 and 100")
            return
        
        print(f"  > Full book: {full_word_count} words")
        limit_words = int(full_word_count * (args.process_limit / 100.0))
        print(f"  > Processing: {args.process_limit}% (~{limit_words} words)")
        
        limited_pages = []
        collected_words = 0
        for p in pages:
            limited_pages.append(p)
            collected_words += count_words(p.text)
            if collected_words >= limit_words:
                break
        pages = limited_pages
        if not pages: pages = [PageContent("")] 

    # Set up prompts based on format and detected content type
    if detected_book_type == "FICTION":
        if output_format == "reading":
            prompt_file = os.path.join("prompts", "prompt_reading_fiction.md")
            default_ext = ".md"
        else:
            prompt_file = os.path.join("prompts", "prompt_tts_fiction.md")
            default_ext = ".txt"
    elif output_format == "tts":
        prompt_file = os.path.join("prompts", "prompt_tts_nonfiction.md")
        default_ext = ".txt"
    elif output_format == "reading":
        prompt_file = os.path.join("prompts", "prompt_reading_nonfiction.md")
        default_ext = ".md"
    else:
        # Fallback to fiction
        prompt_file = os.path.join("prompts", "prompt_tts_fiction.md")
        default_ext = ".txt"

    global_prompt_template = load_prompt_template(prompt_file)

    # Define Title Generation Instruction based on DETECTED CONTENT, not style_key
    if detected_book_type == "NONFICTION":
         print("  > Title Strategy: DESCRIPTIVE (Non-Fiction)")
         title_instruction = "Make the title purely descriptive, professional, and clear. Summarize the main topic."
    else:
         print("  > Title Strategy: SPOILER-FREE (Fiction)")
         title_instruction = "Make the title intriguing and dramatic, but DO NOT reveal major spoilers or plot twists."
    
    # Generate Thematic Focus Instruction
    focus_instruction = ""
    if args.focus:
        theme_list = args.focus.strip()
        print(f"  > Thematic Focus: {theme_list}")
        if detected_book_type == "NONFICTION":
            focus_template = load_prompt_template(os.path.join("prompts", "prompt_focus_nonfiction.md"))
        else:
            focus_template = load_prompt_template(os.path.join("prompts", "prompt_focus_fiction.md"))
        focus_instruction = focus_template.format(theme_list=theme_list)

    # Logging Image Stats
    if enable_vision and filename.lower().endswith('.pdf'):
         total_images = sum(len(p.images) for p in pages)
         print(f"  > Extracted: {total_images} images from {len(pages)} pages")

    original_word_count = count_total_words(pages)
    print(f"\n" + "="*50)
    print("COMPRESSION SETUP")
    print("="*50)
    print(f"  > Input: {original_word_count} words")
    
    # Calculate Target & Ratio
    # Calculate Target & Ratio
    if args.target_minutes:
        base_target_words = int(args.target_minutes * WORDS_PER_MINUTE)
        
        # Scale target based on ACTUAL content fraction
        # Use full_word_count calculated earlier. If 0 (empty book), avoid div by zero.
        if 'full_word_count' in locals() and full_word_count > 0:
             scale_factor = original_word_count / full_word_count
        else:
             scale_factor = 1.0 # Fallback if full count parsing failed or logic skipped
             
        if args.process_limit < 100.0:
             print(f"  > Scaling by: {scale_factor:.4f} (fraction processed)")
             target_total_words = int(base_target_words * scale_factor)
        else:
             target_total_words = base_target_words

        effective_ratio = target_total_words / original_word_count if original_word_count > 0 else 1.0
        print(f"  > Target: {target_total_words} words (~{args.target_minutes:.0f} min @ {WORDS_PER_MINUTE} wpm)")
        print(f"  > Effective Ratio: {effective_ratio:.2f}")
    else:
        target_total_words = int(original_word_count * args.ratio)
        effective_ratio = args.ratio
        print(f"  > Target: {target_total_words} words")
        print(f"  > Ratio: {args.ratio}")
    
    # Validation checks based on effective_ratio
    if effective_ratio < 0.05:
         print(f"WARNING: Extremely low effective ratio ({effective_ratio:.2f}). Significant loss of detail likely.")

    # Calculate chunks based on INPUT size limit (prevents hallucinations) and OUTPUT limit
    num_chunks_by_input = math.ceil(original_word_count / MAX_INPUT_WORDS_PER_CHUNK)  # Keep input manageable
    num_chunks_by_output = math.ceil(target_total_words / SAFE_OUTPUT_WORDS)  # Don't exceed output limit
    num_chunks = max(num_chunks_by_input, num_chunks_by_output)  # Use whichever requires more chunks
    
    avg_input_per_chunk = original_word_count / num_chunks if num_chunks > 0 else original_word_count
    avg_output_per_chunk = target_total_words / num_chunks if num_chunks > 0 else target_total_words
    print(f"\n  > Chunking: {num_chunks} chunks (~{avg_input_per_chunk:.0f} -> ~{avg_output_per_chunk:.0f} words/chunk)")
    
    chunks = split_into_chunks(pages, num_chunks)
    actual_num_chunks = len(chunks)
    print(f"  > Actual chunks created: {actual_num_chunks}")
    
    # Post-process: Check if any chunk would exceed SAFE_OUTPUT_WORDS when processed
    # This can happen with expansion ratios when semantic chunking creates fewer chunks
    needs_resplit = []
    for i, chunk in enumerate(chunks):
        chunk_words = count_words(chunk.text)
        chunk_target = int(chunk_words * effective_ratio)
        if chunk_target > SAFE_OUTPUT_WORDS:
            needs_resplit.append(i)
    
    if needs_resplit:
        print(f"  > Detected {len(needs_resplit)} chunk(s) exceeding safe limit. Applying intelligent split...")
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if i in needs_resplit:
                chunk_words = count_words(chunk.text)
                chunk_target = int(chunk_words * effective_ratio)
                print(f"    - Splitting chunk {i+1} (target {chunk_target} words exceeds {SAFE_OUTPUT_WORDS} limit)")
                # Split at paragraph/sentence boundaries
                sub_chunks = split_chunk_at_boundary(chunk, int(SAFE_OUTPUT_WORDS / effective_ratio))
                print(f"      -> Created {len(sub_chunks)} sub-chunks")
                new_chunks.extend(sub_chunks)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
        actual_num_chunks = len(chunks)
        print(f"  > Final chunk count: {actual_num_chunks}")
    
    # Post-splitting validation: merge tiny chunks with adjacent ones
    # Calculate minimum input based on compression ratio and minimum output requirement
    minimum_input_words = int(MINIMUM_OUTPUT_WORDS_PER_CHUNK / effective_ratio)
    
    tiny_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_words = count_words(chunk.text)
        if chunk_words < minimum_input_words:
            tiny_chunks.append(idx)
    
    if tiny_chunks:
        print(f"  > Detected {len(tiny_chunks)} tiny chunk(s) (< {minimum_input_words} words). Merging with adjacent chunks...")
        merged_chunks = []
        skip_next = False
        
        for idx, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue
            
            chunk_words = count_words(chunk.text)
            if idx in tiny_chunks:
                # Merge with previous chunk if possible, otherwise with next
                if merged_chunks:
                    # Merge with previous
                    prev_chunk = merged_chunks[-1]
                    prev_chunk.text += "\n\n" + chunk.text
                    prev_chunk.images.extend(chunk.images)
                    print(f"    - Merged tiny chunk {idx + 1} ({chunk_words} words) into chunk {len(merged_chunks)}")
                elif idx + 1 < len(chunks):
                    # Merge with next
                    next_chunk = chunks[idx + 1]
                    from collections import namedtuple
                    ChunkInfo = namedtuple('ChunkInfo', ['text', 'images'])
                    merged_chunk = ChunkInfo(
                        text=chunk.text + "\n\n" + next_chunk.text,
                        images=chunk.images + next_chunk.images
                    )
                    merged_chunks.append(merged_chunk)
                    skip_next = True
                    print(f"    - Merged tiny chunk {idx + 1} ({chunk_words} words) with chunk {idx + 2}")
                else:
                    # Last chunk and tiny - keep it anyway (rare edge case)
                    merged_chunks.append(chunk)
            else:
                merged_chunks.append(chunk)
        
        chunks = merged_chunks
        actual_num_chunks = len(chunks)
        print(f"  > Final chunk count after merging: {actual_num_chunks}")
    
    # target_per_chunk is now calculated dynamically per chunk.
    
    compressed_chunks: List[str | None] = [None] * actual_num_chunks
    
    if args.target_minutes:
        action_verb = "Adapting to Time"
        noun_verb = "Time Adaptation"
    elif args.ratio > 1.0:
        action_verb = "Extending"
        noun_verb = "Extension"
    elif args.ratio == 1.0:
        action_verb = "Rewriting"
        noun_verb = "Rewrite"
    else:
        action_verb = "Compressing"
        noun_verb = "Compression"

    print(f"\n" + "="*50)
    print(f"{action_verb.upper()} CHUNKS")
    print("="*50)
    print(f"  > Processing {actual_num_chunks} chunks in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {}
        for i, chunk in enumerate(chunks):
            # Dynamic Target Calculation:
            chunk_words = count_words(chunk.text)
            chunk_target = int(chunk_words * effective_ratio)
            
            # Ensure minimum meaningful output
            chunk_target = max(chunk_target, MINIMUM_OUTPUT_WORDS_PER_CHUNK)
            
            # Log exact stats for every chunk
            print(f"  > Chunk {i+1}: {chunk_words} words -> Target {chunk_target} words")
            
            # Use i+1 as chunk number for numbering
            future = executor.submit(compress_chunk_with_template, chunk, i+1, chunk_target, client, args.model, args.language, global_prompt_template, args.temperature, title_instruction, focus_instruction, args.attempts, output_format)
            future_to_index[future] = i
        
        try:
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    data = future.result()
                    compressed_chunks[i] = data
                    print(f"  > Chunk {i+1}/{actual_num_chunks} completed")
                except (TokenLimitExceededError, EmptyContentError) as e:
                    print(f"\n{e}") # Short message only
                    print("ABORTING...")
                    # Cancel all pending futures
                    executor.shutdown(wait=False, cancel_futures=True)
                    return # Exit process_file immediately (skips write_file)
                except Exception as exc:
                    print(f"\nChunk {i+1} error: {exc}")
                    print("ABORTING due to unexpected error...")
                    # Cancel all pending futures
                    executor.shutdown(wait=False, cancel_futures=True)
                    return # Exit process_file immediately (skips write_file) 
        except Exception:
             pass 
        
    final_text = "\n\n".join(str(c) if c else "" for c in compressed_chunks).strip()
    final_word_count = count_words(final_text)
    
    base_name = os.path.splitext(filename)[0]
    
    # Feature for ElevenReader: Add clean title as first line (Optional)
    if args.add_title:
        readable_title = base_name.replace("_", " ").title()
        final_content = f"{readable_title}\n\n{final_text}"
    else:
        final_content = final_text
    
    output_path = os.path.join("Output", f"{base_name}{default_ext}")
    
    write_file(output_path, final_content)
    
    print(f"\n" + "="*50)
    print(f"{noun_verb.upper()} COMPLETE")
    print("="*50)
    print(f"  > Saved to: {output_path}")
    print(f"  > Final Ratio: {final_word_count / original_word_count:.2f} ({original_word_count} -> {final_word_count} words)")
    estimated_minutes = final_word_count / WORDS_PER_MINUTE
    print(f"  > Estimated Time: ~{estimated_minutes:.1f} min @ {WORDS_PER_MINUTE} wpm")

def main():
    parser = argparse.ArgumentParser(description="AI Book Compressor (via OpenRouter)")
    # Removed input_file and output arguments
    parser.add_argument("-r", "--ratio", type=float, default=0.5, help="Compression ratio (0.001 to 2.0). Default 0.5")
    parser.add_argument("-M", "--target_minutes", type=float, help="Target reading time in minutes (Approx 150 wpm). Overrides --ratio.")
    parser.add_argument("-p", "--process_limit", type=float, default=100.0, help="Percentage of the input file to process (0.1 to 100). Default 100")
    parser.add_argument("-m", "--model", help="OpenRouter Model ID (default: google/gemini-3-flash-preview)", default=MODEL_NAME)
    parser.add_argument("-f", "--format", help="Output format: 'tts' (Text-to-Speech optimized, default), 'reading' (structured for reading)", default="tts")
    parser.add_argument("-l", "--language", help="Output language. Default is 'Slovak'", default="Slovak")
    parser.add_argument("-t", "--temperature", type=float, default=None, help="LLM Temperature (0.0=Strict, 1.0=Creative). Default: Model Default")
    parser.add_argument("-F", "--focus", help="Thematic focus keywords (comma-separated) to prioritize in output. E.g., 'magic,spells' or 'economics,policy'", default="")
    parser.add_argument("-a", "--attempts", type=int, default=2, help="Number of draft attempts per chunk (1-5). Uses adaptive target compensation. Default 2")
    parser.add_argument("-T", "--add_title", action="store_true", help="Prepend the filename as a title at the beginning of the output file. Default is False.")
    
    args = parser.parse_args()
    
    if not (0.001 <= args.ratio <= 2.0):
        print("ERROR: Ratio must be between 0.001 and 2.0")
        return

    # Validate attempts - must be at least 1
    if args.attempts < 1:
        print("ERROR: --attempts must be at least 1")
        return
    args.attempts = min(args.attempts, 5)  # Cap at 5 max

    if args.ratio > 1.0:
         print(f"WARNING: Ratio > 1.0 ({args.ratio}). The AI will attempt to EXPAND/REWRITE the content to be longer than the original.")
         time.sleep(1) 

    if not os.path.exists("Input"):
        os.makedirs("Input")
        print("Created 'Input' directory. Please place your files here and run again.")
        return
        
    if not os.path.exists("Output"):
        os.makedirs("Output")

    try:
        client = setup_client()
    except Exception as e:
        print(f"ERROR: Failed to setup client - {e}")
        return
    
    files = [f for f in os.listdir("Input") if f.lower().endswith(('.pdf', '.epub'))]
    
    if not files:
        print("ERROR: No .pdf or .epub files found in 'Input' directory")
        return

    print(f"\n" + "="*50)
    print(f"BOOK COMPRESSOR")
    print("="*50)
    print(f"  > Found {len(files)} book(s) to process")
    
    for filename in files:
        process_file(filename, args, client)

if __name__ == "__main__":
    main()
