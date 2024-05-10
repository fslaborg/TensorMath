// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace TensorMath

[<AutoOpen>]
module OpBMMExtensions =

    type Tensor with
        /// <summary>Batched matrix product of two tensors. Tensors <paramref name="b" /> must be 3d tensors each containing the same number of matrices. If the tensor is a \(b \times n \times m\) tensor, and <paramref name="b" /> is a \(b \times m \times p\) tensor, the result will be a \(b \times n \times p\) tensor.</summary>
        /// <param name="b">The second tensor.</param>
        member a.bmm(b:Tensor) =
            Shape.checkCanBMM a.shape b.shape |> ignore
            TensorC(a.primalRaw.BMMTT(b.primalRaw))

    type dsharp with
        /// <summary>Batched matrix product of two tensors. Tensors <paramref name="a" /> and  <paramref name="b" /> must be 3d tensors each containing the same number of matrices. If <paramref name="a" /> is a \(b \times n \times m\) tensor, <paramref name="b" /> is a \(b \times m \times p\) tensor, the result will be a \(b \times n \times p\) tensor.</summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        static member bmm(a:Tensor, b:Tensor) = a.bmm(b)
