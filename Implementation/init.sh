#!/bin/bash
git submodule init
git submodule update

cp LeadFL/fltk/core/client.py LeadFL/fltk/core/client_original.py
cp LeadFL-modification/client_modified.py LeadFL/fltk/core/client.py
