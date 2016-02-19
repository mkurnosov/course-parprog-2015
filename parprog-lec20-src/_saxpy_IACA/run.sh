#!/bin/sh

~/opt/iaca-lin64/bin/iaca.sh -64 -arch IVB -analysis LATENCY -o report -graph graph ./saxpy
dot -Tpng ./graph1.dot -o graph1.png


 