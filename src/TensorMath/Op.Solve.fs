// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace TensorMath

[<AutoOpen>]
module OpSolveExtensions =

    type Tensor with
        member a.solve(b:Tensor) =
            let _ = Shape.checkCanSolve a.shape b.shape
            TensorC(a.primalRaw.SolveTT(b.primalRaw))

    type dsharp with
        static member solve(a:Tensor, b:Tensor) = a.solve(b)
