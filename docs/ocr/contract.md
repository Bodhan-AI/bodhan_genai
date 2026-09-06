# The IndicOCR contract

Prompts, block taxonomy and output schema. All of it lives in one module,
`bodhan_genai.ocr.templates.contract`, so the CLI, the engine and any downstream consumer read the
same definitions. Print the live values at any time:

```bash
python -m bodhan_genai.ocr.inference.cli show-contract
```

## Two vocabularies

Mixing these up is the easiest way to break the pipeline, and it fails silently — a block quietly
becomes `Text`.

**Labels** are what IndicDocLayout emits: the 37 education-domain class names in
`ocr.layout.labels.CLASSES`, spelled exactly as they appear there (`Page-number`,
`Sub-section-title`). **Types** are the coarse categories a label maps to; types select the prompt
and gate reconstruction.

Case sensitivity differs by set, deliberately:

| set | compared | why |
| --- | --- | --- |
| `MARGINALIA`, `HEAD_FOOT` | case-**sensitively**, exact class spellings | they gate layout cleanup and must match the model's own strings |
| `OCR_SKIP_LABELS` | case-**insensitively** | the one set a caller is likely to hand-edit |

## Types

Reachable types are exactly what `map_label` can produce. Labels absent from the map fall through
to `Text` by design — `Question`, `Paragraph`, `Answer`, `List`, `MCQ`, `Code`, `Reference` and the
rest all carry prose.

| | types |
| --- | --- |
| **Kept** (reach the output) | `Text` `Title` `SectionHeader` `Table` `Equation` `Caption` `Footnote` `PageHeader` `PageFooter` `PageNumber` |
| **Dropped** (never cropped, never reconstructed) | `Figure` `Picture` |

`KEPT_BLOCK_TYPES` is asserted by test to be exactly the reachable types less `DROP_TYPES`, so a
dead entry or an undocumented one cannot creep in.

`DROP_TYPES` is deliberately narrow: pictorial regions have no text and asking the recognizer to
read one invites hallucination, but page numbers and other margin text *are* transcribed.

## Prompts

| block type | prompt |
| --- | --- |
| `Table` | HTML by default (`colspan`/`rowspan`/`<br/>`); GFM markdown selectable |
| `Equation` | LaTeX only, no surrounding prose |
| everything else | transcription, with math rendered as `$...$` / `$$...$$` |

The text prompt being math-aware is what makes inline-equation dedup safe: a paragraph already
renders its own inline math, so the separate nested equation box would emit it a second time.

## Skipped labels

`OCR_SKIP_LABELS` = `header` `footer` `diagram` `image` `chart` `advertisement`.

Blocks carrying these are never sent to the recognizer, but they are **not deleted**. They keep
their place in both JSON files with `text: ""`, so a consumer can still see that a running header
or a figure was detected, and where.

## Output schema

`<name>.layout.json` and `<name>.json` share one envelope; only the per-block `text` distinguishes
them.

```json
{
  "image": "page.png",
  "width": 800,
  "height": 1273,
  "blocks": [
    {
      "order": 0,
      "label": "Paragraph",
      "type": "Text",
      "bbox_xyxy": [74.5, 406.2, 427.7, 432.6],
      "conf": 0.737,
      "text": "..."
    }
  ]
}
```

| field | meaning |
| --- | --- |
| `order` | reading-order rank, 0-based and gap-free |
| `label` | the raw IndicDocLayout class |
| `type` | the coarse pipeline category |
| `bbox_xyxy` | pixel box `[x0, y0, x1, y1]`, clamped to the page |
| `conf` | detection confidence |
| `text` | transcription; `""` when not sent to the recognizer |

Key order in the JSON is fixed and load-bearing — the regression gate compares output byte for
byte, so `Block.as_record` writes these keys in this order deliberately.

`<name>.md` is the blocks joined in reading order with a blank line between them, bare equations
wrapped in `$$…$$`, and hyphenation across line breaks repaired.
