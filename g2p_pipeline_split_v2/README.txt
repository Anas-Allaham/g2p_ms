Files:
- heteronyms.json           : external IPA lexicon with POS -> pronunciation mapping
- utils.py                  : shared helper functions
- pos_tagger.py             : POS tagging layer (spaCy)
- heteronym_lexicon.py      : loads and resolves heteronyms from JSON
- contextual_g2p.py         : NeMo + POS-aware IPA G2P engine
- main.py                   : example; converts final IPA output to ARPAbet

How to run:
1) Put all files in the same folder
2) Install dependencies:
   pip install spacy
   python -m spacy download en_core_web_sm

   NeMo IpaG2p is optional because the service has a dictionary fallback.
   To install the full (heavy) backend from the repository root:
   pip install -r requirements-nemo.txt
3) Run:
   python main.py
