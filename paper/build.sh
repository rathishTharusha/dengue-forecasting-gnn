#!/usr/bin/env bash
# Build the paper through Docker, without needing `make` on the host.
#
#   ./build.sh              main.pdf (clean version, for submission)
#   ./build.sh highlighted  main-highlighted.pdf (colour-coded contributions)
#   ./build.sh both         both PDFs
#   ./build.sh pages        page count of main.pdf
#   ./build.sh shell        interactive shell in the LaTeX container
#   ./build.sh clean        remove build artefacts
#
# Why this exists alongside the Makefile: `make` is not installed on the Windows
# machines in this team, and it is not in the LaTeX image either, so `make` is
# unavailable on both sides. This script is the portable path. The Makefile is
# kept for teammates on Linux/macOS who already have make; both drive the same
# Docker image and produce the same PDFs.
#
# Override the image with:  LATEX_IMAGE=other:tag ./build.sh
set -euo pipefail

IMAGE="${LATEX_IMAGE:-rathish-latex-env:latest}"
cd "$(dirname "$0")"

# Git Bash on Windows reports MSYS paths (/c/Users/...) that the Docker daemon
# cannot resolve, so use `pwd -W` for a native path where it exists. Linux and
# macOS fall through to plain pwd. MSYS_NO_PATHCONV stops Git Bash rewriting the
# container-side path (/work) into a Windows one.
HOST_DIR="$(pwd -W 2>/dev/null || pwd)"
export MSYS_NO_PATHCONV=1

run() { docker run --rm -v "${HOST_DIR}":/work -w /work "$IMAGE" "$@"; }

require_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: Docker image '$IMAGE' not found." >&2
    echo "  Build it, pull it, or point at another with LATEX_IMAGE=<image>." >&2
    exit 1
  fi
}

# pdflatex -> bibtex -> pdflatex -> pdflatex. The repeats resolve \cite and
# \ref cross-references; a single pass leaves them as bold [?] markers.
latex_cycle() {
  local job="$1"
  run pdflatex -interaction=nonstopmode -halt-on-error "$job"
  run bibtex "$job" || true      # bibtex warns on unused entries; not fatal
  run pdflatex -interaction=nonstopmode -halt-on-error "$job"
  run pdflatex -interaction=nonstopmode -halt-on-error "$job"
}

page_count() { run pdfinfo main.pdf | awk '/^Pages/{print $2}'; }

report_pages() {
  local n
  n="$(page_count)"
  echo
  echo "Total pages: ${n}"
  echo "Limit: 4 pages EXCLUDING references."
  if [ "$n" -gt 5 ]; then
    echo "OVER BUDGET. Cut from Related Work and Experimental Setup first, never Results."
  fi
}

case "${1:-all}" in
  all|main)
    require_image
    latex_cycle main
    report_pages
    ;;
  highlighted)
    require_image
    printf '\\def\\HIGHLIGHTON{1}\\input{main.tex}\n' > main-highlighted.tex
    latex_cycle main-highlighted
    echo "--> main-highlighted.pdf"
    ;;
  both)
    "$0" all
    "$0" highlighted
    ;;
  pages)
    require_image
    [ -f main.pdf ] || latex_cycle main
    report_pages
    ;;
  shell)
    require_image
    docker run --rm -it -v "${HOST_DIR}":/work -w /work "$IMAGE" bash
    ;;
  clean)
    rm -f ./*.aux ./*.log ./*.out ./*.bbl ./*.blg ./*.toc ./*.fdb_latexmk ./*.fls \
          main-highlighted.tex main-highlighted.pdf main.pdf
    echo "cleaned"
    ;;
  *)
    echo "usage: $0 [all|highlighted|both|pages|shell|clean]" >&2
    exit 2
    ;;
esac
