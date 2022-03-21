#!/usr/bin/env wolframscript
(* ::Package:: *)

Needs["NumericalCalculus`"]


CircleTimes = KroneckerProduct;
SuperDagger[x_]:=ConjugateTranspose[x];
{Id,X,Y,Z} = PauliMatrix/@Range[0,3];

beta = {1.1, 3.3};
gamma = {1.5,1.6,1.7};

costFn[beta_,gamma_,printVec_]:=Block[{},
	n = 4;

	H = 0.1*Id\[CircleTimes]Id\[CircleTimes]Id\[CircleTimes]Id + 0.4*Z\[CircleTimes]Id\[CircleTimes]Id\[CircleTimes]Id + 0.4*Id\[CircleTimes]Z\[CircleTimes]Id\[CircleTimes]Id + 0.3*Z\[CircleTimes]Z\[CircleTimes]Id\[CircleTimes]Id + 0.4*Z\[CircleTimes]Id\[CircleTimes]Z\[CircleTimes]Id + 0.5*Z\[CircleTimes]Z\[CircleTimes]Z\[CircleTimes]Z + 0.6*Id\[CircleTimes]Id\[CircleTimes]Z\[CircleTimes]Z;
	(*julia code is little-endian in qubit ordering*)
	ansatz = {X\[CircleTimes]Z\[CircleTimes]X\[CircleTimes]Y, X\[CircleTimes]Id\[CircleTimes]Id\[CircleTimes]Id + Id\[CircleTimes]X\[CircleTimes]Id\[CircleTimes]Id + Id\[CircleTimes]Id\[CircleTimes]X\[CircleTimes]Id + Id\[CircleTimes]Id\[CircleTimes]Id\[CircleTimes]X};

	V[i_]:=MatrixExp[-I gamma[[i]] H];
	U[i_]:=MatrixExp[-I beta[[i]] ansatz[[i]]];
	psi0 = Table[1,{i,1,2^n}];
	psi0 = psi0 / Norm[psi0];

	psiF = (V[3]) . (U[2] . V[2]) . (U[1] . V[1]) . psi0;
	energy = SuperDagger[psiF] . H . psiF//Chop;
	If[printVec,Print[psiF]];
	energy
	]


fn[b1_,b2_,g1_,g2_,g3_]:=costFn[{b1,b2},{g1,g2,g3},False];
gradFn[b1_,b2_,g1_,g2_,g3_]:={
	ND[fn[b1,b2,x,g2,g3],x,g1],
	ND[fn[x,b2,g1,g2,g3],x,b1],
	ND[fn[b1,b2,g1,x,g3],x,g2],
	ND[fn[b1,x,g1,g2,g3],x,b2],
	ND[fn[b1,b2,g1,g2,x],x,g3]
};


costFn[beta,gamma,True]


gradFn[Sequence@@beta,Sequence@@gamma]//Chop
