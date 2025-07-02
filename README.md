# Huddler Neural

A Python module for advanced question detection, answerability classification, and direction detection in multi-party conversations. Designed for use in technical interviews, meetings, and academic presentations, it leverages rule-based and (optionally) BERT-based NLP techniques to determine if a question is directed at a specific user and whether the user should respond.

## Features
- **Question Detection:** Identifies questions in free-form text using spaCy and regex patterns.
- **Answerability Classification:** Classifies questions as factual, opinion, rhetorical, speculative, or clarification, and determines if the user can answer based on context.
- **Direction Detection:** Determines if a question is directed at the user (by name, role, context, or general addressing) using rule-based logic and optional BERT entailment.
- **Comprehensive Test Suite:** Includes a robust test framework with diverse scenarios for validation.

## Requirements
- Python 3.7+
- [spaCy](https://spacy.io/) (`en_core_web_sm` model)
- [transformers](https://huggingface.co/transformers/) and [torch](https://pytorch.org/) (optional, for BERT-based direction detection)
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage
Run the comprehensive test suite (default entry point):
```bash
python huddler_neural.py
```

You can also import and use the main classes in your own code:
```python
from huddler_neural import EnhancedHuddlerPipeline

pipeline = EnhancedHuddlerPipeline(
    user_name="Sarah",
    user_role="software engineer",
    user_context="...",
    meeting_context="...",
    use_bert=False,
    use_bert_direction=True
)
questions, answerability, direction = pipeline.process(
    text="Sarah, can you describe your experience with REST APIs?",
    speaker_id=1,
    user_context="..."
)
```

## Test Cases
The test suite covers:
- Direct name and role-based addressing
- Contextual and follow-up questions
- General and rhetorical questions
- Technical and business expertise matching
- Multi-question messages

## License
MIT License

## Author
- [Your Name Here]

---
*This project is for research and educational purposes. Contributions welcome!*
