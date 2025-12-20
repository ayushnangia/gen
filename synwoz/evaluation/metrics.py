#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for Task-Oriented Dialogues

This module provides detailed implementations and documentation for computing
evaluation metrics on dialogue data, including:

1. PERPLEXITY - Measures language model fluency/naturalness
2. PRECISION/RECALL - Measures entity extraction and response accuracy
3. COHERENCE - Measures dialogue flow and consistency

=============================================================================
PERPLEXITY: MATHEMATICAL FOUNDATION
=============================================================================

Perplexity measures how well a language model predicts a sequence of tokens.
Lower perplexity indicates the model finds the text more "expected" or natural.

FORMULA:
--------
For a sequence of tokens W = (w_1, w_2, ..., w_N):

    PPL(W) = exp( -1/N * sum_{i=1}^{N} log P(w_i | w_1, ..., w_{i-1}) )

Which simplifies to:

    PPL(W) = exp( average cross-entropy loss )

Where:
    - P(w_i | w_1, ..., w_{i-1}) is the probability the model assigns to
      token w_i given all previous tokens
    - N is the number of tokens in the sequence
    - The cross-entropy loss for each token is: -log P(w_i | context)

INTERPRETATION:
---------------
- Perplexity of K means the model is "as confused" as if it had to choose
  uniformly among K options at each step
- Lower is better: PPL=10 means ~10 equally likely choices per token
- Typical ranges:
  - Well-trained LM on in-domain text: 10-50
  - Generic LM on domain-specific text: 50-200
  - Poor model or out-of-domain text: 200+

IMPLEMENTATION DETAILS:
-----------------------
1. We use a pre-trained causal LM (GPT-2 or similar)
2. For each text, we compute the model's loss (cross-entropy)
3. Loss is averaged across all tokens
4. Perplexity = exp(average_loss)

Note: We exclude the first token from loss calculation since it has no
context for prediction.

=============================================================================
PRECISION & RECALL: DIALOGUE EVALUATION
=============================================================================

In task-oriented dialogues, precision and recall measure different aspects:

1. ENTITY-LEVEL PRECISION/RECALL
   - Measures how well the system identifies/mentions correct entities
   - Entities: restaurants, hotels, times, locations, prices, etc.

   Precision = |Predicted Entities ∩ Ground Truth Entities| / |Predicted Entities|
   Recall = |Predicted Entities ∩ Ground Truth Entities| / |Ground Truth Entities|

2. RESPONSE-LEVEL PRECISION/RECALL (using embeddings)
   - Measures semantic similarity between generated and reference responses
   - Uses embedding cosine similarity with a threshold

   For a threshold τ:
   - A response is "correct" if cosine_sim(generated, reference) >= τ

3. SLOT-FILLING PRECISION/RECALL
   - For DST (Dialogue State Tracking) evaluation
   - Compares predicted slot-value pairs against ground truth

=============================================================================
COHERENCE: DIALOGUE FLOW QUALITY
=============================================================================

Coherence measures how well consecutive turns in a dialogue relate to each other.

FORMULA:
--------
For a dialogue with turns T = (t_1, t_2, ..., t_n):

    Coherence = 1/(n-1) * sum_{i=1}^{n-1} cosine_sim(embed(t_i), embed(t_{i+1}))

Where embed() uses a sentence transformer model.

INTERPRETATION:
---------------
- Higher coherence (closer to 1.0) indicates better dialogue flow
- Low coherence may indicate topic jumps or incoherent responses
- Typical ranges for good dialogues: 0.3-0.6

=============================================================================
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

# Lazy imports for optional dependencies
_torch = None
_transformers = None
_sentence_transformers = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers


def _get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        import sentence_transformers
        _sentence_transformers = sentence_transformers
    return _sentence_transformers


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """
    Result container for perplexity computation.

    Attributes:
        perplexity: The computed perplexity value (exp of avg cross-entropy)
        avg_loss: Average cross-entropy loss per token
        total_tokens: Total number of tokens evaluated
        num_sequences: Number of text sequences evaluated
        model_name: Name of the language model used
    """
    perplexity: float
    avg_loss: float
    total_tokens: int
    num_sequences: int
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'perplexity': self.perplexity,
            'avg_loss': self.avg_loss,
            'total_tokens': self.total_tokens,
            'num_sequences': self.num_sequences,
            'model_name': self.model_name
        }


@dataclass
class PrecisionRecallResult:
    """
    Result container for precision/recall computation.

    Attributes:
        precision: Proportion of predicted items that are correct
        recall: Proportion of ground truth items that were predicted
        f1: Harmonic mean of precision and recall
        true_positives: Number of correct predictions
        false_positives: Number of incorrect predictions
        false_negatives: Number of missed ground truth items
    """
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }


def compute_perplexity(
    texts: List[str],
    model_name: str = "gpt2",
    max_length: int = 512,
    batch_size: int = 1,
    device: Optional[str] = None,
    show_progress: bool = True
) -> PerplexityResult:
    """
    Compute perplexity of texts using a pre-trained language model.

    MATHEMATICAL DEFINITION:
    ------------------------
    Perplexity measures how well a probability model predicts a sample.
    For a sequence W = (w_1, ..., w_N):

        PPL(W) = exp( -1/N * Σ log P(w_i | w_<i) )
               = exp( average cross-entropy loss )

    WHERE:
    - P(w_i | w_<i) is the probability of token w_i given previous tokens
    - N is the number of tokens
    - Lower perplexity = model finds text more predictable/natural

    INTERPRETATION:
    ---------------
    - PPL of K means model is "as uncertain" as choosing among K options
    - Good dialogue text: PPL 20-80 with GPT-2
    - Random/incoherent text: PPL 200+

    Args:
        texts: List of text strings to evaluate
        model_name: HuggingFace model name (default: "gpt2")
                   Options: "gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing (currently processes one at a time)
        device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
        show_progress: Whether to show progress bar

    Returns:
        PerplexityResult with perplexity, avg_loss, and token counts

    Example:
        >>> texts = ["Hello, how can I help you today?",
        ...          "I'd like to book a table for two at 7pm."]
        >>> result = compute_perplexity(texts, model_name="gpt2")
        >>> print(f"Perplexity: {result.perplexity:.2f}")
        Perplexity: 45.23

    Notes:
        - The first token is excluded from loss calculation (no context)
        - Texts shorter than 2 tokens are skipped
        - Uses teacher forcing (ground truth as input at each step)
    """
    torch = _get_torch()
    transformers = _get_transformers()

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Computing perplexity using {model_name} on {device}")
    logger.info(f"Processing {len(texts)} text sequences")

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_tokens = 0
    num_sequences = 0

    # Progress bar setup
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(texts, desc="Computing perplexity")
    else:
        iterator = texts

    with torch.no_grad():
        for text in iterator:
            # Tokenize the text
            # Note: We use return_tensors='pt' to get PyTorch tensors
            encodings = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=False  # No padding for single sequences
            )
            input_ids = encodings.input_ids.to(device)

            # Skip very short sequences (need at least 2 tokens for loss)
            # The first token has no context, so we need at least 2
            if input_ids.shape[1] < 2:
                logger.debug(f"Skipping short sequence: {input_ids.shape[1]} tokens")
                continue

            # Forward pass with labels=input_ids computes cross-entropy loss
            # The loss is computed for predicting token i from tokens 0..i-1
            # HuggingFace internally shifts labels to compute next-token prediction
            outputs = model(input_ids, labels=input_ids)

            # outputs.loss is the mean cross-entropy loss over all tokens
            # We need to track total loss and tokens for proper averaging
            loss = outputs.loss.item()

            # Number of tokens for which loss is computed
            # (sequence length - 1, since first token has no prediction)
            num_tokens = input_ids.shape[1] - 1

            # Accumulate weighted loss
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            num_sequences += 1

    # Compute average loss and perplexity
    if total_tokens == 0:
        logger.warning("No valid tokens found for perplexity calculation")
        return PerplexityResult(
            perplexity=float('inf'),
            avg_loss=float('inf'),
            total_tokens=0,
            num_sequences=0,
            model_name=model_name
        )

    avg_loss = total_loss / total_tokens

    # PERPLEXITY FORMULA: PPL = exp(average cross-entropy loss)
    # Cross-entropy loss is: -log P(w_i | context)
    # So: PPL = exp(-1/N * Σ log P(w_i | context)) = exp(avg_loss)
    perplexity = math.exp(avg_loss)

    logger.info(f"Perplexity: {perplexity:.2f} (avg_loss: {avg_loss:.4f})")
    logger.info(f"Evaluated {total_tokens} tokens across {num_sequences} sequences")

    return PerplexityResult(
        perplexity=perplexity,
        avg_loss=avg_loss,
        total_tokens=total_tokens,
        num_sequences=num_sequences,
        model_name=model_name
    )


def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> Set[str]:
    """
    Extract entities from dialogue text using pattern matching.

    This is a simplified entity extraction for common TOD entities.
    For production use, consider using spaCy NER or a trained model.

    Args:
        text: Text to extract entities from
        entity_types: List of entity types to extract (default: all)
                     Options: "time", "date", "price", "location", "phone"

    Returns:
        Set of extracted entity strings (normalized to lowercase)
    """
    entities = set()
    text_lower = text.lower()

    # Time patterns: 7pm, 7:30pm, 19:00, etc.
    if entity_types is None or "time" in entity_types:
        time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b',
            r'\b\d{1,2}\s*(?:am|pm)\b',
            r'\b(?:noon|midnight)\b'
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

    # Date patterns: tomorrow, monday, 15th, etc.
    if entity_types is None or "date" in entity_types:
        date_patterns = [
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:today|tomorrow|tonight)\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

    # Price patterns: $50, 50 dollars, etc.
    if entity_types is None or "price" in entity_types:
        price_patterns = [
            r'\$\d+(?:\.\d{2})?',
            r'\b\d+\s*(?:dollars?|pounds?|euros?)\b',
            r'£\d+(?:\.\d{2})?',
            r'€\d+(?:\.\d{2})?'
        ]
        for pattern in price_patterns:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

    # Phone patterns
    if entity_types is None or "phone" in entity_types:
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\b\d{10,11}\b'
        ]
        for pattern in phone_patterns:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

    # Number of people: 2 people, party of 4, etc.
    if entity_types is None or "party_size" in entity_types:
        party_patterns = [
            r'\b(\d+)\s*(?:people|persons?|guests?)\b',
            r'\bparty\s*of\s*(\d+)\b',
            r'\bfor\s*(\d+)\b',
            r'\btable\s*for\s*(\d+)\b'
        ]
        for pattern in party_patterns:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                entities.add(f"{m} people")

    return entities


def compute_entity_precision_recall(
    predicted_texts: List[str],
    reference_texts: List[str],
    entity_types: Optional[List[str]] = None
) -> PrecisionRecallResult:
    """
    Compute entity-level precision and recall between predicted and reference texts.

    MATHEMATICAL DEFINITION:
    ------------------------
    Given:
    - P = set of entities in predicted texts
    - R = set of entities in reference texts

    Precision = |P ∩ R| / |P|    (What fraction of predictions are correct?)
    Recall = |P ∩ R| / |R|       (What fraction of ground truth was found?)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        predicted_texts: List of generated/predicted dialogue texts
        reference_texts: List of ground truth dialogue texts
        entity_types: Types of entities to extract (default: all)

    Returns:
        PrecisionRecallResult with precision, recall, F1, and counts

    Example:
        >>> predicted = ["I found a table for 2 at 7pm at The Italian Place"]
        >>> reference = ["Book a table for 2 people at 7:00pm at Italian Restaurant"]
        >>> result = compute_entity_precision_recall(predicted, reference)
        >>> print(f"Precision: {result.precision:.2f}, Recall: {result.recall:.2f}")
    """
    all_predicted_entities = set()
    all_reference_entities = set()

    for pred_text, ref_text in zip(predicted_texts, reference_texts):
        pred_entities = extract_entities(pred_text, entity_types)
        ref_entities = extract_entities(ref_text, entity_types)
        all_predicted_entities.update(pred_entities)
        all_reference_entities.update(ref_entities)

    # Compute true positives, false positives, false negatives
    true_positives = len(all_predicted_entities & all_reference_entities)
    false_positives = len(all_predicted_entities - all_reference_entities)
    false_negatives = len(all_reference_entities - all_predicted_entities)

    # Compute precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # Compute F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PrecisionRecallResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def compute_response_precision_recall(
    predicted_responses: List[str],
    reference_responses: List[str],
    similarity_threshold: float = 0.8,
    model_name: str = "all-MiniLM-L6-v2"
) -> PrecisionRecallResult:
    """
    Compute response-level precision/recall using semantic similarity.

    METHODOLOGY:
    ------------
    1. Embed all responses using a sentence transformer
    2. For each predicted response, check if it's "close enough" to any reference
    3. A response is considered a "match" if cosine_sim >= threshold

    This is useful when exact matching is too strict but you want to measure
    whether the generated responses convey similar information.

    Args:
        predicted_responses: List of generated responses
        reference_responses: List of ground truth responses
        similarity_threshold: Cosine similarity threshold for considering a match
        model_name: Sentence transformer model to use

    Returns:
        PrecisionRecallResult with precision, recall, F1, and counts
    """
    sentence_transformers = _get_sentence_transformers()

    model = sentence_transformers.SentenceTransformer(model_name)

    # Encode all responses
    pred_embeddings = model.encode(predicted_responses, convert_to_numpy=True)
    ref_embeddings = model.encode(reference_responses, convert_to_numpy=True)

    # Normalize for cosine similarity
    pred_embeddings = pred_embeddings / np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
    ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(pred_embeddings, ref_embeddings.T)

    # Count matches based on threshold
    matched_predictions = set()
    matched_references = set()

    for i, pred in enumerate(predicted_responses):
        for j, ref in enumerate(reference_responses):
            if similarity_matrix[i, j] >= similarity_threshold:
                matched_predictions.add(i)
                matched_references.add(j)

    true_positives = len(matched_predictions)
    false_positives = len(predicted_responses) - true_positives
    false_negatives = len(reference_responses) - len(matched_references)

    precision = true_positives / len(predicted_responses) if predicted_responses else 0.0
    recall = len(matched_references) / len(reference_responses) if reference_responses else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PrecisionRecallResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def compute_coherence(
    dialogues: List[List[str]],
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Compute dialogue coherence using sentence embeddings.

    MATHEMATICAL DEFINITION:
    ------------------------
    For a dialogue with turns T = (t_1, t_2, ..., t_n):

        Coherence = 1/(n-1) * Σ_{i=1}^{n-1} cosine_sim(embed(t_i), embed(t_{i+1}))

    This measures how semantically related consecutive turns are.
    High coherence indicates smooth topic flow and relevant responses.

    INTERPRETATION:
    ---------------
    - 0.0-0.2: Low coherence, possibly random or unrelated turns
    - 0.2-0.4: Moderate coherence, typical for diverse topic dialogues
    - 0.4-0.6: Good coherence, focused task-oriented dialogues
    - 0.6+: High coherence, very focused single-topic conversations

    Args:
        dialogues: List of dialogues, where each dialogue is a list of turn strings
        model_name: Sentence transformer model to use

    Returns:
        Average coherence score across all dialogues

    Example:
        >>> dialogues = [
        ...     ["Hi, I need a restaurant", "What cuisine do you prefer?", "Italian please"],
        ...     ["Book a hotel", "For how many nights?", "Two nights"]
        ... ]
        >>> coherence = compute_coherence(dialogues)
        >>> print(f"Coherence: {coherence:.4f}")
    """
    sentence_transformers = _get_sentence_transformers()

    model = sentence_transformers.SentenceTransformer(model_name)

    coherence_scores = []

    for dialogue in dialogues:
        if len(dialogue) < 2:
            continue

        # Encode all turns
        embeddings = model.encode(dialogue, convert_to_numpy=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute consecutive similarities
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            coherence_scores.append(sim)

    return np.mean(coherence_scores) if coherence_scores else 0.0


class DialogueEvaluator:
    """
    Comprehensive evaluator for task-oriented dialogues.

    This class provides a unified interface for computing multiple
    evaluation metrics on dialogue data.

    Example:
        >>> evaluator = DialogueEvaluator(device="cuda")
        >>> dialogues = load_dialogues("generated.jsonl")
        >>> results = evaluator.evaluate_all(dialogues)
        >>> print(results)
    """

    def __init__(
        self,
        perplexity_model: str = "gpt2",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the evaluator.

        Args:
            perplexity_model: Model for perplexity calculation
            embedding_model: Model for coherence and similarity
            device: Device to use (auto-detected if None)
            max_length: Maximum sequence length
        """
        self.perplexity_model = perplexity_model
        self.embedding_model = embedding_model
        self.device = device
        self.max_length = max_length

        # Lazy load models
        self._ppl_model = None
        self._ppl_tokenizer = None
        self._embed_model = None

    def format_dialogue_text(self, dialogue: Dict) -> str:
        """Format a dialogue dictionary as text for evaluation."""
        parts = []
        for turn in dialogue.get('turns', []):
            utterance = turn.get('utterance', '').strip()
            response = turn.get('assistant_response', '').strip()
            if utterance:
                parts.append(f"User: {utterance}")
            if response:
                parts.append(f"Assistant: {response}")
        return '\n'.join(parts)

    def format_dialogue_turns(self, dialogue: Dict) -> List[str]:
        """Extract individual turn strings from a dialogue."""
        turns = []
        for turn in dialogue.get('turns', []):
            utterance = turn.get('utterance', '').strip()
            response = turn.get('assistant_response', '').strip()
            if utterance:
                turns.append(utterance)
            if response:
                turns.append(response)
        return turns

    def evaluate_perplexity(
        self,
        dialogues: List[Dict],
        sample_size: Optional[int] = None
    ) -> PerplexityResult:
        """
        Evaluate perplexity of dialogues.

        Args:
            dialogues: List of dialogue dictionaries
            sample_size: Optional sample size (for large datasets)

        Returns:
            PerplexityResult with computed metrics
        """
        texts = [self.format_dialogue_text(d) for d in dialogues]

        if sample_size and len(texts) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(texts), sample_size, replace=False)
            texts = [texts[i] for i in indices]

        return compute_perplexity(
            texts,
            model_name=self.perplexity_model,
            max_length=self.max_length,
            device=self.device
        )

    def evaluate_coherence(self, dialogues: List[Dict]) -> float:
        """
        Evaluate coherence of dialogues.

        Args:
            dialogues: List of dialogue dictionaries

        Returns:
            Average coherence score
        """
        formatted_dialogues = [self.format_dialogue_turns(d) for d in dialogues]
        return compute_coherence(formatted_dialogues, self.embedding_model)

    def evaluate_all(
        self,
        dialogues: List[Dict],
        reference_dialogues: Optional[List[Dict]] = None,
        perplexity_sample_size: Optional[int] = 100
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics on dialogues.

        Args:
            dialogues: List of generated dialogue dictionaries
            reference_dialogues: Optional list of reference dialogues for P/R
            perplexity_sample_size: Sample size for perplexity (None for all)

        Returns:
            Dictionary with all computed metrics
        """
        results = {}

        # Perplexity
        logger.info("Computing perplexity...")
        ppl_result = self.evaluate_perplexity(dialogues, perplexity_sample_size)
        results['perplexity'] = ppl_result.to_dict()

        # Coherence
        logger.info("Computing coherence...")
        coherence = self.evaluate_coherence(dialogues)
        results['coherence'] = coherence

        # Entity precision/recall if references provided
        if reference_dialogues:
            logger.info("Computing entity precision/recall...")
            pred_texts = [self.format_dialogue_text(d) for d in dialogues]
            ref_texts = [self.format_dialogue_text(d) for d in reference_dialogues]

            entity_result = compute_entity_precision_recall(pred_texts, ref_texts)
            results['entity_precision_recall'] = entity_result.to_dict()

        return results


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate task-oriented dialogues with multiple metrics"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to JSONL file with dialogues'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--perplexity-model',
        type=str,
        default='gpt2',
        help='Model for perplexity (default: gpt2)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Sample size for perplexity calculation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, mps, cpu)'
    )

    args = parser.parse_args()

    # Load dialogues
    dialogues = []
    with open(args.input_file, 'r') as f:
        for line in f:
            dialogues.append(json.loads(line))

    print(f"Loaded {len(dialogues)} dialogues")

    # Evaluate
    evaluator = DialogueEvaluator(
        perplexity_model=args.perplexity_model,
        device=args.device
    )

    results = evaluator.evaluate_all(
        dialogues,
        perplexity_sample_size=args.sample_size
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Perplexity: {results['perplexity']['perplexity']:.2f}")
    print(f"Average Loss: {results['perplexity']['avg_loss']:.4f}")
    print(f"Coherence: {results['coherence']:.4f}")

    # Save if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
