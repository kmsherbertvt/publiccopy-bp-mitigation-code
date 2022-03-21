#!/usr/bin/env wolframscript
(* ::Package:: *)

CircleTimes = KroneckerProduct;
SuperDagger[x_]:=ConjugateTranspose[x];
{Id,X,Y,Z} = PauliMatrix/@Range[0,3];


X\[CircleTimes]Y + Z\[CircleTimes]Z//MatrixForm


Y\[CircleTimes]X + Z\[CircleTimes]Z//MatrixForm
