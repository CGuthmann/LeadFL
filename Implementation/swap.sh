#!/bin/bash

if [ ! -f LeadFL/fltk/core/client_original.py ]; then
    echo "First run init!"
    exit
fi

if [ "$1" = "original" ]; then 
    cp LeadFL/fltk/core/client_original.py LeadFL/fltk/core/client.py 
    echo "LeadFL now uses the original client"

elif [ "$1" = "modified" ]; then 
    cp LeadFL-modification/client_modified.py LeadFL/fltk/core/client.py
    echo "LeadFL now uses the modified client"

else
    echo "Usage: ./swap.sh (original|modified)"
fi