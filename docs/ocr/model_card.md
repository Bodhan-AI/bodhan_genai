---
language: [en, as, bn, brx, doi, gu, hi, kn, ks, kok, mai, ml, mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur]
license: apache-2.0
pipeline_tag: image-to-text
tags: [ocr, document-parsing, layout-analysis, reading-order, indic, vision-language-model, qwen, rt-detr]
---

# IndicOCR

**Document parsing for English and 22 Indian languages, printed and handwritten.** A page image
in; reading-ordered Markdown out, with math as LaTeX and tables as HTML or Markdown, plus
per-block JSON.

The published card, with sample pages and diagrams, is at
[bodhan-ai/indic-ocr](https://huggingface.co/bodhan-ai/indic-ocr). This page is the
same content for readers working inside the repo.

IndicOCR reads a document page and returns its text in reading order. It is a modular,
two-stage parser: **IndicDocLayout** detects the blocks on the page and orders them, and
**IndicBlockOCR** transcribes the textual blocks. The two stages communicate through a structured
JSON file, so either stage can be used independently or replaced with another implementation.

## Model Summary

| | IndicDocLayout | IndicBlockOCR |
| --- | --- | --- |
| **Role** | Layout detection + reading order | Block-level text recognition |
| **Architecture** | PP-DocLayoutV3 / RT-DETR | Qwen3.5-0.8B |
| **Parameters** | 33 M | 0.8 B |
| **Precision** | fp32 | bf16 |
| **In the Hub repo** | `weights/layout` (133 MB) | `weights/ocr` (1.7 GB) |
| **Runtime** | PyTorch | vLLM |
| **Output** | Layout JSON | Markdown + block JSON |

IndicBlockOCR uses the **Sarvam-30B tokenizer**, with a vocabulary designed to cover Indian
scripts. IndicDocLayout is a fine-tune of PP-DocLayoutV3/RT-DETR, trained with a 37-class
taxonomy designed for education-domain documents.

## Supported languages

**Printed** page recognition is supported across English and the 22 constitutionally recognised
Indian languages: Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani,
Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil,
Telugu, Urdu.

**Handwriting** recognition currently supports English and 12 Indian languages: Hindi, Bengali,
Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, and Urdu.

Handwriting quality is still a work in progress, particularly across different writing styles. We
are working on improving recognition and extending support to additional languages.

## Usage

Inside this repo, see [the package README](https://github.com/Bodhan-AI/bodhan_genai/blob/main/src/bodhan_genai/ocr/README.md)
for install and the Python API, and [end-to-end](end-to-end.md) for a walkthrough. To use the weights directly from the
Hub, follow the install and inference sections of the
[model card](https://huggingface.co/bodhan-ai/indic-ocr).

## Output

`parser.parse("page.png")` returns the page metadata and its blocks in reading order:

```json
{
  "image": "sample1.png",
  "width": 800,
  "height": 1273,
  "blocks": [
    {"order": 0, "label": "Header", "type": "PageHeader",
     "bbox_xyxy": [345.6, 51.7, 437.1, 114.7], "conf": 0.6, "text": ""},
    {"order": 1, "label": "Page-number", "type": "PageNumber",
     "bbox_xyxy": [367.8, 78.9, 413.2, 107.4], "conf": 0.747, "text": "229"},
    {"order": 2, "label": "Paragraph", "type": "Text",
     "bbox_xyxy": [77.9, 121.3, 711.8, 199.4], "conf": 0.863,
     "text": "Thus we see that, if we can prove that twice the L.H.S. of (30) ..."}
  ]
}
```

| field | meaning |
| --- | --- |
| `order` | reading-order rank, 0-based and gap-free |
| `label` | the raw IndicDocLayout class (37-class taxonomy) |
| `type` | coarse pipeline category: `Text`, `Table`, `Equation`, `Title`, ... |
| `bbox_xyxy` | pixel box `[x0, y0, x1, y1]` |
| `conf` | detection confidence |
| `text` | transcription; `""` for blocks not sent to the recognizer |

**Note:** Figures, charts, advertisements, running headers, and footers are not sent through the
recognizer by default. They remain in the JSON with `text: ""`, so you can see what was detected
and where. Page numbers and other margin text such as folios are transcribed.

### Table format

Tables come back as HTML by default. Choose the format when you construct the parser:

```python
IndicOCR()  # HTML (default)
IndicOCR(recognizer_config=RecognizerConfig(table_format=TableFormat.MARKDOWN))
```

Or on the command line, with `--table-format markdown`.

## Performance

### olmOCR-Bench (English)

| System | Overall | ArXiv math | Baseline | Headers/Footers | Long/tiny | Multi-col | Old scans | Old-scan math | Tables |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicOCR** | **82.9** | 84.5 | 99.2 | 98.3 | 90.3 | 73.5 | 47.3 | 80.8 | 89.5 |
| Sarvam-OCR | 84.3 | 86.5 | 99.6 | 96.3 | 91.0 | 82.2 | 49.8 | 81.0 | 88.3 |
| PaddleOCR-VL-1.6 | 78.7 | 85.1 | 98.7 | 95.8 | 75.1 | 84.1 | 39.0 | 68.8 | 82.9 |

Re-measurement under the shipped configuration is pending. The 82.9 run used
`dedup_mode=text_only`, `min_px_side=256` and tables as Markdown; the shipped defaults are
`dedup_mode=both` with tables as HTML.

### Printed recognition performance

Needs reevaluation.

### Handwriting recognition performance

Needs reevaluation.

### IndicDLP (layout)

Per-source detection and reading-order scores for IndicDocLayout ship alongside the checkpoint in
`test_metrics.json`: mAP@50 0.58 to 0.73 and reading-order Kendall tau 0.91 to 0.98 across seven
sources, covering printed textbooks, magazines, national archives and four handwriting sets.

## Limitations

Reading order remains a challenge for **complex, multi-column layouts**. Handwriting recognition
is also still being improved, particularly across different writing styles and writing
characteristics.

We are also extending handwriting support to additional Indic languages.

One page per image: rendering PDF pages to images is up to the caller. Engine startup dominates
single-page use, so pass a directory where you can.

## Hardware

Linux with one GPU. Latency and throughput numbers to follow.

## License

This model and repository are released under the **Apache 2.0 License**.

The release incorporates components distributed under Apache 2.0, including PP-DocLayoutV3,
Qwen3.5, and the Sarvam-30B tokenizer. See the repository license and the corresponding upstream
licenses for the applicable terms and attribution requirements.

## Citation

```bibtex
@misc{indicocr2026,
  title  = {IndicOCR: Multilingual Document Parsing for English and 22 Indian Languages},
  author = {Bodhan.AI},
  year   = {2026},
  url    = {https://huggingface.co/bodhan-ai/indic-ocr}
}
```
