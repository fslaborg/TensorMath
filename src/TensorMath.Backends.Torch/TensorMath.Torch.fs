﻿namespace TensorMath

open TensorMath
open TensorMath.Backends.Torch
open TorchSharp

[<AutoOpen>]
module TorchExtensions =

    type dsharp with

        /// <summary>
        /// Creates a new TensorMath tensor from the torch tensor.
        /// </summary>
        static member fromTorch(tt: torch.Tensor) =
            Tensor.ofRawTensor(TorchRawTensor(tt))

    type Tensor with
        /// <summary>
        /// Converts the primal of a tensor to a torch tensor.
        /// </summary>
        /// <remarks>
        /// If the tensor does not use the Torch backend an exception is raised.
        ///
        /// Note that this operation takes the primal of the tensor. This means
        /// code that converts to Torch tensors will not be differentiable using
        /// TensorMath differentiation capabilities.
        /// </remarks>
        member t.toTorch() =
            match t.primalRaw with
            | :? TorchRawTensor as trt -> trt.TorchTensor
            | _ -> failwith $"toTorch: the input is not a TensorMath.Backends.Torch tensor, its backend is {t.backend}"

