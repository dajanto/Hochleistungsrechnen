#!/usr/bin/env gnuplot

set term pdf;
set termopt enhanced

# Datei, aus der gelesen wird
in_file="laufzeiten-interlines-raw.txt"

# Lade Eingabedatei in datablock $data (hackiger Hack)
set table $data
   plot in_file using 1:2 with table
unset table

# Speichere Wert für 1 Thread in first_time
#first_time=$data[12]

# stats liest eine Menge Infos aus $data und speichert sie in Variablen STATS_*
#stats $data nooutput

# Maximale Anzahl an Punkten wird als Anzahl Records aus stats gesetzt
#max_tries = STATS_records

# Falls in der ersten Spalte der größte Wert ist, setze unteres Limit der x-Achse
# auf 1, ansonsten auf 0.
#ymin = STATS_index_max == 0 ? 1 : 0
#ymin = 0
# Falls wir irgendwo superlinearen Speedup haben, setze ymax dementsprechend
#ymax = ceil(first_time / STATS_min) 
#ymax = ymax > max_tries ? ymax : max_tries

# Setze Grenzen von x- und y-Achse
#set xrange [1 : max_tries]
#set yrange [ymin : ymax]

# Setze Position der Achsen-Tics
#set xtics 1,1,max_tries
#set ytics ymin,1,ymax

#stats in_file using 2 nooutput

set xlabel "Anzahl Interlines"
set ylabel "Zeit pro Interline in Sekunden"

# Legende Oben Links
set key left top

# Ausgabedatei
set output "interline-linear-scale-with-expected.pdf"

f(x) = 0.05 * x * log(x)

# $0 ist die nullbasierte Zeilennummer
# $0 + 1 ist die 1-basierte Zeilennummer (= Anzahl Threads)
# $1 ist der Wert in der ersten (und einzigen) Spalte
# first_time / $1 ist der Speedup
# Plotte die Anzahl Threads gegen den Speedup, und den Optimalen Speedup
plot in_file with linespoints lt rgb "#FF0000" title "partdiff-openmp-zeilen Interline Skalierung", \
   f(x) with lines lt rgb "#AAAAAA" title "Erwartete Interline Skalierung"
