#!/usr/bin/env wolframscript
(* ::Package:: *)

Needs["NumericalCalculus`"]


CircleTimes = KroneckerProduct;
SuperDagger[x_]:=ConjugateTranspose[x];
{Id,X,Y,Z} = PauliMatrix/@Range[0,3];

beta = {1.1};
gamma = {1.5,1.4};

costFn[beta_,gamma_,printVec_]:=Block[{},
	n = 2;

	H = 0.4*Z\[CircleTimes]Id + 0.5*Id\[CircleTimes]Z;
	(*julia code is little-endian in qubit ordering*)
	ansatz = {X\[CircleTimes]Id + Id\[CircleTimes]X};

	V[i_]:=MatrixExp[-I gamma[[i]] H];
	U[i_]:=MatrixExp[-I beta[[i]] ansatz[[i]]];
	psi0 = Table[1,{i,1,2^n}];
	psi0 = psi0 / Norm[psi0];

	psiF = (V[2]) . (U[1] . V[1]) . psi0;
	energy = SuperDagger[psiF] . H . psiF//Chop;
	If[printVec,Print[psiF]];
	energy
	]


fn[b1_,g1_,g2_]:=costFn[{b1},{g1,g2},False];
gradFn[b1_,g1_,g2_]:={
	ND[fn[x,g1,g2],x,b1],
	ND[fn[b1,x,g2],x,g1],
	ND[fn[b1,g1,x],x,g2]
};


costFn[beta,gamma,True]


gradFn[Sequence@@beta,Sequence@@gamma]//Chop
