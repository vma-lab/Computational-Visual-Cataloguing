# ScriptSight (part of CVC)

ScriptSight is the interactive exploration tool of **CVC (Computational Visual Cataloguing)**. It is **one of three components** in the parent repository, alongside [`../page_detection/`](../page_detection) and [`../word_pipeline/`](../word_pipeline). See the top-level [`../README.md`](../README.md) for the overall CVC system and the paper citation.

ScriptSight can be used to explore and sort collections of document-page images using their AI-generated 'computational visual catalogues'. Multiple visual attributes, such as text orientation, used writing implements (pencil versus ink), and text colour, can be filtered concurrently to isolate pages matching the specified criteria. Matching pages can be previewed as thumbnails with optional overlays indicating detected text regions, and saved as organised, date-stamped folders for review or further processing.

The source code is also released as a standalone record on the Universität Hamburg research data repository: https://www.fdr.uni-hamburg.de/record/17616. A test sample of a generated computational visual catalogue, with 30 notebook pages from Rainer Maria Rilke (Deutsches Literaturarchiv Marbach, A: Rilke-Archiv Gernsbach), is available at https://www.fdr.uni-hamburg.de/record/17614.

## Expected JSON structure

ScriptSight ingests catalogues produced by the word pipeline as COCO-style JSON files with the following structure:

![Structure of the CVC JSON expected by ScriptSight](./Computational%20Visual%20Catalogue.png)

## Configuration

ScriptSight persists settings in a `config.json` file located next to the executable. The configuration includes a boolean `show_tool_labels` option which controls whether the writing-tool (e.g. pencil, ink) labels are drawn on thumbnails and overlay previews. Set this value to `false` to hide the labels.
