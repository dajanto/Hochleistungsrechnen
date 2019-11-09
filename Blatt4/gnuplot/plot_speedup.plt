#!/usr/bin/env gnuplot

set term pdf;
set termopt enhanced

# Datei, aus der gelesen wird
in_file="laufzeiten.txt"

# Lade Eingabedatei in datablock $data (hackiger Hack)
set table $data
   plot in_file using 1 with table
unset table

# Speichere Wert für 1 Thread in first_time
first_time=$data[1]

# stats liest eine Menge Infos aus $data und speichert sie in Variablen STATS_*
stats $data nooutput

# Maximale Anzahl Threads wird als Anzahl Records aus stats gesetzt
max_threads = STATS_records

# Falls in der ersten Spalte der größte Wert ist, setze unteres Limit der x-Achse
# auf 1, ansonsten auf 0.
ymin = STATS_index_max == 0 ? 1 : 0

# Falls wir irgendwo superlinearen Speedup haben, setze ymax dementsprechend
ymax = ceil(first_time / STATS_min) 
ymax = ymax > max_threads ? ymax : max_threads

# Setze Grenzen von x- und y-Achse
set xrange [1 : max_threads]
set yrange [ymin : ymax]

# Setze Position der Achsen-Tics
set xtics 1,1,max_threads
set ytics ymin,1,ymax

set xlabel "Anzahl Threads"
set ylabel "Speedup"

# Legende Oben Links
set key left top

# Ausgabedatei
set output "speedup.pdf"

f(x) = x

# $0 ist die nullbasierte Zeilennummer
# $0 + 1 ist die 1-basierte Zeilennummer (= Anzahl Threads)
# $1 ist der Wert in der ersten (und einzigen) Spalte
# first_time / $1 ist der Speedup
# Plotte die Anzahl Threads gegen den Speedup, und den Optimalen Speedup
plot "$data" using ($0+1):(first_time/$1) with linespoints lt rgb "#FF0000" title "Speedup", \
      "$data" using ($0+1):(791.25/$1) with linespoints lt rgb "#0000FF" title "Speedup zum sequentiellen", \
     f(x) with lines lt rgb "#AAAAAA" title "Optimaler Speedup";
