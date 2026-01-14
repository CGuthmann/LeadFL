(TeX-add-style-hook
 "eg_refs"
 (lambda ()
   (setq TeX-command-extra-options
         "-shell-escape")
   (LaTeX-add-bibitems
    "art"
    "art2"))
 :bibtex)

