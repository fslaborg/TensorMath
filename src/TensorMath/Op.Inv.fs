// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace TensorMath

[<AutoOpen>]
module OpInvExtensions =

    type Tensor with
        member a.inv() =
            Shape.checkCanInvert a.shape
            TensorC(a.primalRaw.InverseT())

    type dsharp with
        static member inv(a:Tensor) = a.inv()
