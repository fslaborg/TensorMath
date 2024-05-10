// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace TensorMath

[<AutoOpen>]
module OpDetExtensions =

    type Tensor with
        member a.det() =
            Shape.checkCanDet a.shape
            TensorC(a.primalRaw.DetT())

    type dsharp with
        static member det(a:Tensor) = a.det()
