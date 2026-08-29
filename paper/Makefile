# CS3631 Phase 2 short paper
#
#   make              -> main.pdf              clean version (conference submission)
#   make highlighted  -> main-highlighted.pdf  colour-coded contributions (grading)
#   make both         -> both PDFs
#   make pages        -> print page count of main.pdf  (MUST be <= 4 + references)
#   make clean        -> remove build artefacts
#
# Both PDFs are built from the SAME sources. Never fork the text.

LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIB   = bibtex

.PHONY: all both highlighted pages clean

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard sections/*.tex) $(wildcard figures/*)
	$(LATEX) main
	-$(BIB) main
	$(LATEX) main
	$(LATEX) main

# Injects \HIGHLIGHTON before the class, flipping every \cA..\cF into a
# highlight and switching on the \credit margin notes.
highlighted: main.tex refs.bib $(wildcard sections/*.tex)
	@printf '\\def\\HIGHLIGHTON{1}\\input{main.tex}\n' > main-highlighted.tex
	$(LATEX) main-highlighted
	-$(BIB) main-highlighted
	$(LATEX) main-highlighted
	$(LATEX) main-highlighted
	@echo "--> main-highlighted.pdf"

both: main.pdf highlighted

# Run this EVERY time you add a paragraph. Catching the overflow on Saturday
# is how papers get gutted at midnight.
pages: main.pdf
	@echo "Total pages: $$(pdfinfo main.pdf | awk '/^Pages/{print $$2}')"
	@echo "Limit: 4 pages EXCLUDING references."

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.fdb_latexmk *.fls \
	      main-highlighted.tex main-highlighted.pdf main.pdf
