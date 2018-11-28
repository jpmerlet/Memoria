# modelo para upit y pcpsp
param NBLOCKS;
param NPERIODS;
param NDESTINATIONS;
param NRESOURCE_SIDE_CONSTRAINTS;
param DISCOUNT_RATE;
set setBL = 0..NBLOCKS-1;
set setT = 0..NPERIODS-1;
set setSIDE = 0..NRESOURCE_SIDE_CONSTRAINTS-1;
set setDEST = 0..NDESTINATIONS-1;
set setUPIT within setBL;

param RESOURCE_CONSTRAINT_UB_LIMITS {setSIDE,setT};
param RESOURCE_CONSTRAINT_LB_LIMITS {setSIDE,setT};
param RESOURCE_CONSTRAINT_COEFFICIENTS {setBL, setSIDE, setDEST} default 0;
#param OBJECTIVE_FUNCTION {setBL,setDEST};
param OBJECTIVE_FUNCTION_UPIT {setBL};
param OBJECTIVE_FUNCTION_PCPSP {setBL, setDEST};
set PREC{setBL};

var x_upit{setBL} binary;
var x_pcpsp{setUPIT,setT} binary;
var Y{setUPIT,setDEST,setT} >=0, <=1;

maximize profit_upit: sum{b in setBL} OBJECTIVE_FUNCTION_UPIT[b]*x_upit[b];
maximize profit_pcpsp: sum{t in setT} sum {d in setDEST} sum{b in setUPIT} (1.0/(1.0 + DISCOUNT_RATE))^t * OBJECTIVE_FUNCTION_PCPSP[b,d] * Y[b,d,t];

# restricciones para upit
subject to precedence {b in setBL, j in PREC[b]}: x_upit[b] <= x_upit[j];

# restricciones para pcpsp
subject to Resource { r in setSIDE, t in setT}:
		RESOURCE_CONSTRAINT_LB_LIMITS[r,t] <= sum{b in setUPIT} sum {d in setDEST} RESOURCE_CONSTRAINT_COEFFICIENTS[b,r,d] * Y[b,d,t] <= RESOURCE_CONSTRAINT_UB_LIMITS[r,t];

subject to CliqueBlock {b in setUPIT}:
		sum{t in setT} x_pcpsp[b,t] <= 1;

subject to SumDest {b in setUPIT, t in setT}:
		sum {d in setDEST} Y[b,d,t] = x_pcpsp[b,t];

subject to Precedence {b in setUPIT, i in PREC[b], t in setT}:
    sum{s in 0..t} x_pcpsp[b,s] <= sum{s in 0..t} x_pcpsp[i,s];
problem UPIT: x_upit, precedence, profit_upit;
problem PCPSP: x_pcpsp, Y, Resource, CliqueBlock, SumDest, Precedence, profit_pcpsp;