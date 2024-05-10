// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace TensorMath

[<AutoOpen>]
module OpAvgPoolExtensions =

    type Tensor with
        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.avgpool1d(kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let stride = defaultArg stride kernelSize
            let padding = defaultArg padding 0
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Shape.checkCanAvgpool1d a.dtype a.shape kernelSize stride padding |> ignore
            TensorC(a.primalRaw.AvgPool1D(kernelSize, stride, padding(* , ceil_mode, count_include_pad *)))

        member internal a.avgpoolReverse1d(originalInput:Tensor, kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let stride = defaultArg stride kernelSize
            let padding = defaultArg padding 0
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            TensorC(a.primalRaw.AvgPoolReverse1D(originalInput.primalRaw, kernelSize, stride, padding(* , ceil_mode, count_include_pad *)))

        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        member a.avgpool2d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Shape.checkCanAvgpool2d a.dtype a.shape kernelSizes strides paddings  |> ignore
            TensorC(a.primalRaw.AvgPool2D(kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *)))

        member internal a.avgpoolReverse2d(originalInput:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            TensorC(a.primalRaw.AvgPoolReverse2D(originalInput.primalRaw, kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *)))

        /// <summary>Applies a 3D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        member a.avgpool3d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Shape.checkCanAvgpool3d a.dtype a.shape kernelSizes strides paddings  |> ignore
            TensorC(a.primalRaw.AvgPool3D(kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *)))

        member internal a.avgpoolReverse3d(originalInput:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            TensorC(a.primalRaw.AvgPoolReverse3D(originalInput.primalRaw, kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *)))

    type dsharp with
        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member avgpool1d(input: Tensor, kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpool2d(kernelSize=kernelSize, ?stride=stride, ?padding=padding(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Applies a 2D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        static member avgpool2d(input: Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpool2d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Applies a 2D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        static member avgpool3d(input: Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpool3d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

