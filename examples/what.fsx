#!/usr/bin/env -S dotnet fsi

#I "../tests/TensorMath.Tests/bin/Debug/net8.0"
#r "TensorMath.dll"
#r "TensorMath.Backends.Reference.dll"
#r "TensorMath.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
#r "nuget: TorchSharp-cpu"
// #r "nuget: TorchSharp-cuda-linux, 0.96.5"
//#r "nuget: TorchSharp-cuda-windows" // #r "nuget: TorchSharp-cuda-windows, 0.96.5"
// Option B: you can use a local libtorch installation
// System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open TensorMath


dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(1)

let t1 = dsharp.tensor [|1.; 2.; 3.; 4.; |]

t1 * t1

