#!/usr/bin/env wolframscript
(* ::Package:: *)

Needs["NumericalCalculus`"]


CircleTimes = KroneckerProduct;
SuperDagger[x_]:=ConjugateTranspose[x];
{Id,X,Y,Z} = PauliMatrix/@Range[0,3];

beta = {1.1};

costFn[x_,printVec_]:=Block[{},
	n = 2;
	H = X\[CircleTimes]Id + Id\[CircleTimes]X;
	(*julia code is little-endian in qubit ordering*)
	(*ansatz = {X\[CircleTimes]Id + Id\[CircleTimes]X};*)
	ansatz = {0.4*Z\[CircleTimes]Id + 0.5*Id\[CircleTimes]Z};

	U[i_]:=MatrixExp[-I x[[i]] ansatz[[i]]];
	(*psi0 = Table[1,{i,1,2^n}];*)
	psi0 = {1,2,3,4};
	psi0 = psi0 / Norm[psi0];

	psiF = (U[1]) . psi0;
	energy = SuperDagger[psiF] . H . psiF//Chop;
	If[printVec,Print[psiF]];
	energy
	]


fn[x1_]:=costFn[{x1},False];
gradFn[x1_]:={
	ND[fn[x],x,x1]
};


costFn[beta,True]


gradFn[Sequence@@beta(*,Sequence@@gamma*)]//Chop
