"""bodhan_genai.ocr.layout -- the IndicDocLayout model (PP-DocLayoutV3 + reading-order head).

``labels`` is pure Python and safe to import anywhere; the modelling and inference modules pull
in torch and transformers, so import those only when a detector is actually being built.
"""
