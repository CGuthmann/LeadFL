(TeX-add-style-hook
 "example"
 (lambda ()
   (setq TeX-command-extra-options
         "-shell-escape")
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("ICEbeamerTUMCD" "english" "aspectratio=1610" "10pt" "helvet" "nicetitles")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "style=apa")))
   (TeX-run-style-hooks
    "latex2e"
    "ICEbeamerTUMCD"
    "ICEbeamerTUMCD10"
    "biblatex")
   (TeX-add-symbols
    "PersonTitel"
    "PersonVorname"
    "PersonNachname"
    "PersonStadt"
    "PersonAdresse"
    "PersonTelefon"
    "PersonEmail"
    "PersonWebseite"
    "FakultaetName"
    "LehrstuhlName"
    "Datum")
   (LaTeX-add-bibliographies
    "eg_refs"))
 :latex)

