## Archivo (grupo05-mcf-original.mod) ##
param n;
set V:= {1..n};
set A within V cross V;
param n_d;
param s{1..n_d}; 
param t{1..n_d}; 
param d{1..n_d}; 
param loss{1..n_d};
param c{A}; 
param u{A};

set K:= {1..n_d};
var x{(i,j,k) in V cross V cross K: (i,j) in A} >= 0;

minimize MCF: sum{(i,j) in A} c[i,j]*sum {k in K} x[i,j,k];

subject to respetar_cap {(i,j) in A}: sum {k in K}x[i,j,k] <= u[i,j];
subject to flujo_cero_en_nodo {k in K, v in (V diff {s[k],t[k]})}: sum{(v,j) in A} x[v,j,k] - sum{(j,v) in A} x[j,v,k]= 0;
subject to demanda_que_sale {k in K}: sum{(s[k],j) in A} x[s[k],j,k] - sum{(i,s[k]) in A} x[i,s[k],k] = d[k];
subject to demanda_que_entra {k in K}: sum{(t[k],j) in A} x[t[k],j,k] - sum{(i,t[k]) in A} x[i,t[k],k] = -d[k];

