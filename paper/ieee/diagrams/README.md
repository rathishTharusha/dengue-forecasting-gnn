# Editable paper diagrams

Open `architecture.drawio` or `evaluation_protocol.drawio` in draw.io / diagrams.net using File > Open From > Device. These are uncompressed native mxGraph documents with editable text, rounded boxes and orthogonal arrows. They do not contain embedded bitmap figures.

The paper uses their vector PDF exports in `../figures/`. Export the supplied sources without installing draw.io:

```text
python paper/ieee/scripts/drawio_diagrams.py
```

The normal `make_figures.py` entrypoint also exports these diagrams. Existing `.drawio` files are read, not overwritten. Source coordinates use 100 units per inch; export sizes are 7.16 x 2.55 inches and 3.5 x 2 inches, with 9-point Times New Roman text. Connectors use dark 1.15-point strokes and larger filled arrowheads; the protocol boxes have explicit gaps for visible arrow shafts. The lightweight exporter supports the shapes, colors, text and edge points used by these two documents. If adding other draw.io shape types, export from draw.io itself and preserve the physical size and embedded fonts, or extend the exporter. Regenerate and rebuild the LaTeX manuscript after edits.

Color, spacing and grouping are presentation choices only. Solid paths represent inference; dashed training and pending-confirmation paths retain the scientific distinctions in the captions. The forecast-start partition does not remove the documented target overlap.
