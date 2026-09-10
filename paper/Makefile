# CS3631 Phase 2 short paper
#
#   make              -> main.pdf              clean version (conference submission)
#   make highlighted  -> main-highlighted.pdf  colour-coded contributions (grading)
#   make both         -> both PDFs
#   make pages        -> print page count of main.pdf  (MUST be <= 4 + references)
#   make shell        -> interactive shell in the LaTeX container, for debugging
#   make clean        -> remove build artefacts
#
# Both PDFs are built from the SAME sources. Never fork the text.
#
# ---------------------------------------------------------------------------
# TeX runs in Docker. Nobody on the team installs TeX Live locally, and the
# distribution differences between six machines are exactly the kind of thing
# that produces a PDF that builds for one person and not the rest.
#
#   Image:   rathish-latex-env:latest  (TeX Live 2022, includes acmart)
#   Escape:  make NATIVE=1             use a local pdflatex instead
#
# The pwd dance is for Windows. Git Bash reports MSYS paths (/c/Users/...) that
# the Docker daemon cannot resolve, so `pwd -W` is used where available to get a
# native path, and MSYS_NO_PATHCONV stops Git Bash rewriting the container-side
# path. On Linux and macOS `pwd -W` fails and plain `pwd` is used.
# ---------------------------------------------------------------------------

DOCKER_IMAGE ?= rathish-latex-env:latest
HOST_DIR     := $(shell pwd -W 2>/dev/null || pwd)
DOCKER_RUN    = MSYS_NO_PATHCONV=1 docker run --rm -v "$(HOST_DIR)":/work -w /work $(DOCKER_IMAGE)

ifdef NATIVE
  LATEX   = pdflatex -interaction=nonstopmode -halt-on-error
  BIB     = bibtex
  PDFINFO = pdfinfo
else
  LATEX   = $(DOCKER_RUN) pdflatex -interaction=nonstopmode -halt-on-error
  BIB     = $(DOCKER_RUN) bibtex
  PDFINFO = $(DOCKER_RUN) pdfinfo
endif

.PHONY: all both highlighted pages shell clean check-docker

all: main.pdf

# Fails with a useful message rather than a wall of Docker output.
check-docker:
ifndef NATIVE
	@docker image inspect $(DOCKER_IMAGE) >/dev/null 2>&1 || { \
	  echo "ERROR: image $(DOCKER_IMAGE) not found."; \
	  echo "  Build or pull it, or use a local TeX Live with:  make NATIVE=1"; \
	  exit 1; }
endif

main.pdf: check-docker main.tex refs.bib $(wildcard sections/*.tex) $(wildcard figures/*)
	$(LATEX) main
	-$(BIB) main
	$(LATEX) main
	$(LATEX) main

# Injects \HIGHLIGHTON before the class, flipping every \cA..\cF into a
# highlight and switching on the \credit margin notes.
highlighted: check-docker main.tex refs.bib $(wildcard sections/*.tex)
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
	@echo "Total pages: $$($(PDFINFO) main.pdf | awk '/^Pages/{print $$2}')"
	@echo "Limit: 4 pages EXCLUDING references."

# For chasing a LaTeX error interactively without rebuilding the whole document.
shell: check-docker
	$(DOCKER_RUN) bash

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.fdb_latexmk *.fls \
	      main-highlighted.tex main-highlighted.pdf main.pdf
