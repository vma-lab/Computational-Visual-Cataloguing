# Computational Visual Cataloguing (CVC)

**CVC (Computational Visual Cataloguing)** is a computer-vision framework for cataloguing handwritten heritage at scale. It detects content-bearing pages in digitised notebook scans, localises every word across diverse languages, and classifies each word's visual attributes (colour, orientation, writing implement), recording the result in structured, searchable catalogue fields. The framework is designed to learn robustly from densely written manuscript pages and limited human annotations, reflecting the natural challenges of historical handwritten collections (multilingual content, varied orientations and colours, diverse implements, irregular layouts).

CVC is intended to complement, not replace, expert cataloguing. By exposing page-anchored visual attributes as first-class queryable metadata, it helps libraries, museums and archives make large-scale digitised holdings discoverable. The framework is developed and evaluated on the notebooks of **Rainer Maria Rilke** (Deutsches Literaturarchiv Marbach, A: Rilke-Archiv Gernsbach), a culturally significant multilingual corpus.

## Repository layout

This repository bundles the three components of CVC, each in its own folder with its own README:

1. [`page_detection/`](./page_detection): detects the primary content-bearing pages in raw notebook scans.
2. [`word_pipeline/`](./word_pipeline): localises every word on each page and classifies its colour, orientation and writing implement, then writes a structured catalogue; includes a final visualisation stage.
3. [`scriptsight/`](./scriptsight): interactive GUI for filtering, previewing and exporting the resulting catalogues.

A typical end-to-end run is `page_detection/` then `word_pipeline/` then `scriptsight/`, but each component can also be used independently. See each folder's README for setup, usage and per-component details.

## Data, models and catalogues

- **DLA-RMR annotated dataset**: [https://cloud.uni-hamburg.de/s/M4Xx7QswjtgjLjD] (will be moved to [https://www.fdr.uni-hamburg.de/record/18097] after updating the current version)
- **Generated CVC catalogue for Rilke's notebooks**: https://www.fdr.uni-hamburg.de/record/17811
- **Sample catalogue with 30 test pages** (used by ScriptSight demos): https://www.fdr.uni-hamburg.de/record/17614
- **Short ScriptSight demo video**: https://www.csmc.uni-hamburg.de/publications/software/scriptsight.html
- **ScriptSight standalone source release**: https://www.fdr.uni-hamburg.de/record/17616
- **Trained model checkpoints**: https://cloud.uni-hamburg.de/s/q8qZBjFZ7TPSjr3. Also see the per-component READMEs for download links and placement instructions.

## Citation

Citation details will be added here after publication.

## Authors

| Author | Affiliation | Email |
|---|---|---|
| Hussein Mohammed (corresponding author) | Visual Manuscript Analysis Lab (VMA), Centre for the Study of Manuscript Cultures (CSMC), Universität Hamburg | hussein.adnan.mohammed@uni-hamburg.de |
| Quang-Vinh Dang | VMA / CSMC, Universität Hamburg | vinh.dang@uni-hamburg.de |
| Anna Kinder | German Literature Archive (DLA), Marbach | Anna.Kinder@dla-marbach.de |
| Sandra Richter | German Literature Archive (DLA), Marbach | Sandra.Richter@dla-marbach.de |


## Acknowledgements

This work was funded by the **Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)** under Germany's Excellence Strategy, Cluster of Excellence 2176 *Understanding Written Artefacts: Material, Interaction and Transmission in Manuscript Cultures*, project no. 390893796. The research was conducted within the **Centre for the Study of Manuscript Cultures (CSMC)** at Universität Hamburg.

We also thank our colleagues at CSMC, José Maksimczuk and Konrad Hirschler, for insightful discussions, and Hui Xu for help with data preparation.

## License

CVC is released under the **GNU Affero General Public License v3.0**. See [`LICENSE`](./LICENSE) for the full text.
