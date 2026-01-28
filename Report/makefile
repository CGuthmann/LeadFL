# holds the source file
src ?= paper

builddir ?= build
builddirDvi ?= build/dvi
basedir := $(shell pwd)
searchdirs := ".:$(basedir)/:~/texmf/mylib/:"

cmdEngine ?= lualatex
cmdEngineDvi ?= latex
cmdbib ?= bibtex
cmdviewer ?= zathura
cmdpresenter ?= dspdfviewer -t 0 -w 0 -s 0 -p 1

BEEP ?= 0



.PHONY: all pdf dvi bib show showdvi clean_pdf clean_dvi clean present gitignore builddirs beep

all: pdf

pdf: builddirs clean_pdf
	@echo '### Making pdf ###'
	export TEXINPUTS=$(searchdirs) ;\
	$(cmdEngine) -synctex=1 -interaction=nonstopmode --output-directory=$(builddir) $(src).tex
	@$(MAKE) beep

dvi: builddirs clean_dvi
	@echo '### Making dvi ###'
	export TEXINPUTS=$(searchdirs) ;\
	$(cmdEngineDvi) -synctex=1 -interaction=nonstopmode --output-directory=$(builddirDvi) $(src).tex ; true
	dvipdf $(builddirDvi)/$(src).dvi $(builddirDvi)/$(src).dvi.pdf &
	@$(MAKE) beep

bib:
	@( \
		export BIBINPUTS=$(searchdirs) && \
		export BSTINPUTS=$(searchdirs) && \
		cd $(builddir) && \
		if [ -f $(src).aux ]; then \
			echo '### Bib for pdf ###'; \
			$(cmdbib) -terse $(src).aux; \
			echo ' '; \
		fi)
	@( \
		export BIBINPUTS=$(searchdirs) && \
		export BSTINPUTS=$(searchdirs) && \
		cd $(builddirDvi) && \
		if [ -f $(src).aux ]; then \
			echo '### Bib for dvi ###' && \
			$(cmdbib) -terse $(src).aux; \
		fi)

show:
	@echo '### Showing pdf ###'
	$(cmdviewer) $(builddir)/$(src).pdf >/dev/null 2>&1 &

# convert dvi to pdf and show it
showdvi:
	@echo '### Showing dvi ###'
	$(cmdviewer) $(builddirDvi)/$(src).dvi  >/dev/null 2>&1 &

present:
	@echo '### Presenting pdf ###'
	$(cmdpresenter) $(builddir)/$(src).pdf  >/dev/null 2>&1 &

clean_pdf:
	@echo '### Deleting old pdf ###'
	rm -f $(builddir)/$(src).pdf

clean_dvi:
	@echo '### Deleting old dvi ###'
	rm -f $(builddirDvi)/$(src).dvi $(builddirDvi)/$(src).dvi.pdf

clean: clean_pdf clean_dvi
	@echo '### Cleaning ###'
	(if [ -d $(builddir) ]; then \
		cd $(builddir) && \
		rm -f *.aux *.toc *.log *.out *~ *.snm *.nav *.blg *.bbl .gitignore $(src).synctex.gz *.lof *.lot *.exp; \
	fi)
	(if [ -d $(builddirDvi) ]; then \
		cd $(builddirDvi) && \
		rm -f *.aux *.toc *.log *.out *~ *.snm *.nav *.blg *.bbl .gitignore $(src).synctex.gz *.lof *.lot *.exp; \
	fi)
	(if [ -d $(builddirDvi) ]; then rmdir $(builddirDvi); fi)
	(if [ -d $(builddir) ]; then rmdir $(builddir); fi)

gitignore:
	@if [ ! -f .gitignore ] ; then \
		echo '### Creating gitignore file. ###' ; \
		echo -e '$(builddir)/\n$(builddirDvi)/' > .gitignore ; \
	fi

builddirs:
	@echo '### Creating build dirs ###'
	@mkdir -p $(builddir)
	@mkdir -p $(builddirDvi)
	@$(MAKE) gitignore

beep:
	@if [ $(BEEP) -gt 0 ] ; then \
		echo '### Beep! ###'; \
		beep $(BEEP) & \
	fi
