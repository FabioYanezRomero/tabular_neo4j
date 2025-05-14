"""
Utility functions for language detection and validation.
"""
from Tabular_to_Neo4j.utils.logging_config import get_logger
from typing import List, Dict, Tuple, Optional
from langdetect import detect, LangDetectException
from langdetect.detector_factory import DetectorFactory

# Set seed for reproducibility

# Configure logging
logger = get_logger(__name__)

DetectorFactory.seed = 42

# Configure logging


# ISO language code mapping (add more as needed)
ISO_LANGUAGE_CODES = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'nl': 'dutch',
    'ru': 'russian',
    'ja': 'japanese',
    'zh-cn': 'chinese',
    'zh-tw': 'chinese',
    'ar': 'arabic',
    'hi': 'hindi',
    'ko': 'korean',
    'tr': 'turkish',
    'pl': 'polish',
    'sv': 'swedish',
    'fi': 'finnish',
    'da': 'danish',
    'no': 'norwegian',
    'cs': 'czech',
    'hu': 'hungarian',
    'el': 'greek',
    'he': 'hebrew',
    'th': 'thai',
    'vi': 'vietnamese',
    'id': 'indonesian',
    'ms': 'malay',
    'ro': 'romanian',
    'uk': 'ukrainian',
    'bg': 'bulgarian',
    'sk': 'slovak',
    'lt': 'lithuanian',
    'lv': 'latvian',
    'et': 'estonian',
    'hr': 'croatian',
    'sr': 'serbian',
    'sl': 'slovenian',
    'mk': 'macedonian',
    'sq': 'albanian',
    'bs': 'bosnian',
    'ca': 'catalan',
    'eu': 'basque',
    'gl': 'galician',
    'fa': 'persian',
    'ur': 'urdu',
    'bn': 'bengali',
    'ta': 'tamil',
    'te': 'telugu',
    'mr': 'marathi',
    'gu': 'gujarati',
    'kn': 'kannada',
    'ml': 'malayalam',
    'si': 'sinhala',
    'ne': 'nepali',
    'pa': 'punjabi',
    'my': 'burmese',
    'km': 'khmer',
    'lo': 'lao',
    'mn': 'mongolian',
    'jv': 'javanese',
    'sw': 'swahili',
    'zu': 'zulu',
    'xh': 'xhosa',
    'af': 'afrikaans',
    'fy': 'frisian',
    'cy': 'welsh',
    'gd': 'scottish gaelic',
    'ga': 'irish',
    'is': 'icelandic',
    'mt': 'maltese',
    'la': 'latin'
}

def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of a given text.
    
    Args:
        text (str): The text to detect language for
        
    Returns:
        Tuple[str, float]: ISO language code and confidence score
    """
    if not text or len(text.strip()) < 3:
        logger.warning(f"Text too short for language detection: '{text}'")
        return "en", 0.5  # Default to English with low confidence
        
    try:
        # Detect language
        lang_code = detect(text)
        # langdetect doesn't provide confidence, so we use a fixed value
        return lang_code, 0.8  # Reasonable confidence for successful detection
    except LangDetectException as e:
        logger.warning(f"Language detection failed for text: '{text}'. Error: {str(e)}")
        return "en", 0.5  # Default to English with low confidence

def normalize_language_name(language: str) -> str:
    """
    Normalize language name to a standard format.
    
    Args:
        language (str): Language name or code
        
    Returns:
        str: Normalized language name
    """
    # Convert to lowercase
    language = language.lower().strip()
    
    # Check if it's an ISO code
    if language in ISO_LANGUAGE_CODES:
        return ISO_LANGUAGE_CODES[language]
    
    # Check if it's a language name that matches a value in our mapping
    for code, name in ISO_LANGUAGE_CODES.items():
        if language == name:
            return name
    
    # Return as is if we can't normalize
    return language

def are_languages_matching(lang1: str, lang2: str) -> bool:
    """
    Check if two language specifications refer to the same language.
    
    Args:
        lang1 (str): First language name or code
        lang2 (str): Second language name or code
        
    Returns:
        bool: True if they refer to the same language
    """
    norm_lang1 = normalize_language_name(lang1)
    norm_lang2 = normalize_language_name(lang2)
    
    return norm_lang1 == norm_lang2

def verify_header_language(headers: List[str], target_language: str) -> Dict:
    """
    Verify if headers are in the specified target language.
    
    Args:
        headers (List[str]): List of header strings
        target_language (str): Target language name or code
        
    Returns:
        Dict: Dictionary with verification results
        {
            "is_in_target_language": bool,
            "detected_languages": Dict[str, str],  # header -> detected language
            "non_matching_headers": List[str]
        }
    """
    result = {
        "is_in_target_language": True,
        "detected_languages": {},
        "non_matching_headers": []
    }
    
    norm_target = normalize_language_name(target_language)
    
    # Combine all headers for better language detection
    combined_text = " ".join(headers)
    primary_lang = detect_language(combined_text)
    
    if primary_lang:
        norm_primary = normalize_language_name(primary_lang)
        result["primary_detected_language"] = norm_primary
        
        # If the primary language doesn't match target, we already know the result
        if not are_languages_matching(norm_primary, norm_target):
            result["is_in_target_language"] = False
    
    # Check individual headers
    for header in headers:
        # Skip very short headers as they're unreliable for detection
        if len(header) < 4:
            result["detected_languages"][header] = "too_short"
            continue
            
        lang = detect_language(header)
        if lang:
            norm_lang = normalize_language_name(lang)
            result["detected_languages"][header] = norm_lang
            
            if not are_languages_matching(norm_lang, norm_target):
                result["is_in_target_language"] = False
                result["non_matching_headers"].append(header)
        else:
            # If we can't detect, we'll assume it's okay
            result["detected_languages"][header] = "unknown"
    
    return result
