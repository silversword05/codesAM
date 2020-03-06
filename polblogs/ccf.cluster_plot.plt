#
# Polblog Directed graph - clustering coefficient. G(1490, 19090). Average clustering: 0.0210  OpenTriads: 1818150 (0.9961)  ClosedTriads: 7186 (0.0039) (Fri Mar  6 02:53:59 2020)
#

set title "Polblog Directed graph - clustering coefficient. G(1490, 19090). Average clustering: 0.0210  OpenTriads: 1818150 (0.9961)  ClosedTriads: 7186 (0.0039)"
set key bottom right
set logscale xy 10
set format x "10^{%L}"
set mxtics 10
set format y "10^{%L}"
set mytics 10
set grid
set xlabel "Node degree"
set ylabel "Average clustering coefficient"
set tics scale 2
set terminal png font arial 10 size 1000,800
set output 'ccf.cluster_plot.png'
plot 	"ccf.cluster_plot.tab" using 1:2 title "" with linespoints pt 6
