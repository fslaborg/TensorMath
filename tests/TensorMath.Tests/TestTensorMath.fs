// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open TensorMath
open TensorMath.Compose


[<TestFixture>]
type TestTensorMath () =

    let rosenbrock (x:Tensor) = 
        let x, y = x[0], x[1]
        (1. - x)**2 + 100. * (y - x**2)**2
    let rosenbrockGrad (x:Tensor) = 
        let x, y = x[0], x[1]
        dsharp.tensor([-2*(1-x)-400*x*(-(x**2) + y); 200*(-(x**2) + y)])
    let rosenbrockHessian (x:Tensor) = 
        let x, y = x[0], x[1]
        dsharp.tensor([[2.+1200.*x*x-400.*y, -400.*x],[-400.*x, 200.*dsharp.one()]])

    let fscalarscalar (x:Tensor) = dsharp.sin x
    let fscalarscalarDiff (x:Tensor) = dsharp.cos x

    let fscalarvect3 (x:Tensor) = dsharp.stack([sin x; exp x; cos x])
    let fscalarvect3Diff (x:Tensor) = dsharp.stack([cos x; exp x; -sin x])
    let fscalarvect3Diff2 (x:Tensor) = dsharp.stack([-sin x; exp x; -cos x])
    let fscalarvect3Diff3 (x:Tensor) = dsharp.stack([-cos x; exp x; sin x])

    let fvect2vect2 (x:Tensor) = 
        let x, y = x[0], x[1]
        dsharp.stack([x*x*y; 5*x+sin y])
    let fvect2vect2Jacobian (x:Tensor) = 
        let x, y = x[0], x[1]
        dsharp.tensor([[2*x*y; x*x];[dsharp.tensor(5.); cos y]])

    let fvect3vect2 (x:Tensor) = 
        let x, y, z = x[0], x[1], x[2]
        dsharp.stack([x*y+2*y*z;2*x*y*y*z])
    let fvect3vect2Jacobian (x:Tensor) = 
        let x, y, z = x[0], x[1], x[2]
        dsharp.tensor([[y;x+2*z;2*y];[2*y*y*z;4*x*y*z;2*x*y*y]])

    let fvect3vect3 (x:Tensor) = 
        let r, theta, phi = x[0], x[1], x[2]
        dsharp.stack([r*(sin phi)*(cos theta); r*(sin phi)*(sin theta); r*cos phi])
    let fvect3vect3Jacobian (x:Tensor) = 
        let r, theta, phi = x[0], x[1], x[2]
        dsharp.tensor([[(sin phi)*(cos theta); -r*(sin phi)*(sin theta); r*(cos phi)*(cos theta)];[(sin phi)*(sin theta); r*(sin phi)*(cos theta); r*(cos phi)*(sin theta)];[cos phi; dsharp.zero(); -r*sin phi]])

    let fvect3vect4 (x:Tensor) =
        let y1, y2, y3, y4 = x[0], 5*x[2], 4*x[1]*x[1]-2*x[2],x[2]*sin x[0]
        dsharp.stack([y1;y2;y3;y4])
    let fvect3vect4Jacobian (x:Tensor) =
        let z, o = dsharp.zero(), dsharp.one()
        dsharp.tensor([[o,z,z],[z,z,5*o],[z,8*x[1],-2*o],[x[2]*cos x[0],z,sin x[0]]])

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestZero () =
        let t = dsharp.zero(dtype=Int32)
        let tCorrect = dsharp.tensor(0)
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestZeros () =
        let t = dsharp.zeros([2;3], dtype=Int32)
        let tCorrect = dsharp.tensor([[0,0,0],[0,0,0]])
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestOne () =
        let t = dsharp.one(dtype=Int32)
        let tCorrect = dsharp.tensor(1)
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestOnes () =
        let t = dsharp.ones([2;3], dtype=Int32)
        let tCorrect = dsharp.tensor([[1,1,1],[1,1,1]])
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestRand () =
        let t = dsharp.rand([1000])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(0.5)
        let tStddev = t.std()
        let tStddevCorrect = dsharp.tensor(1./12.) |> dsharp.sqrt
        Assert.That(tMeanCorrect.allclose(tMean, 0.1))
        Assert.That(tStddevCorrect.allclose(tStddev, 0.1))

    [<Test>]
    member this.TestRandn () =
        let t = dsharp.randn([1000])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(0.)
        let tStddev = t.std()
        let tStddevCorrect = dsharp.tensor(1.)
        Assert.That(tMeanCorrect.allclose(tMean, 0.1, 0.1))
        Assert.That(tStddevCorrect.allclose(tStddev, 0.1, 0.1))

    [<Test>]
    member this.TestArange () =
        let t1 = dsharp.arange(5.)
        let t1Correct = dsharp.tensor([0.,1.,2.,3.,4.])
        let t2 = dsharp.arange(startVal=1., endVal=4.)
        let t2Correct = dsharp.tensor([1.,2.,3.])
        let t3 = dsharp.arange(startVal=1., endVal=2.5, step=0.5)
        let t3Correct = dsharp.tensor([1.,1.5,2.])
        Assert.CheckEqual(t1Correct, t1)
        Assert.CheckEqual(t2Correct, t2)
        Assert.CheckEqual(t3Correct, t3)


    [<Test>]
    member this.TestSeed () =
        for combo in Combos.FloatingPointExcept16s do
            printfn "%A" (combo.device, combo.backend, combo.dtype)
            use _holder = dsharp.useConfig(combo.dtype, combo.device, combo.backend)
            dsharp.seed(123)
            let t = combo.randint(0,10,[25])
            dsharp.seed(123)
            let t2 = combo.randint(0,10,[25])
            Assert.CheckEqual(t, t2)

    [<Test>]
    member this.TestSlice () =
        let t = dsharp.tensor([1, 2, 3])
        let tSlice = t |> dsharp.slice([0])
        let tSliceCorrect = t[0]
        Assert.CheckEqual(tSliceCorrect, tSlice)

    member _.TestCanConfigure () =
        
        // Backup the current config before the test to restore in the end
        let configBefore = dsharp.config()

        // Default reference backend with CPU
        let device = Device.Default
        dsharp.config(device=Device.CPU)
        Assert.CheckEqual(Device.CPU, Device.Default)
        dsharp.config(device=device)

        // Torch with default backend (CPU)
        let backend = Backend.Default
        dsharp.config(backend=Backend.Torch)
        Assert.CheckEqual(Backend.Torch, Backend.Default)
        dsharp.config(backend=backend)

        // Default reference backend with "int32"
        let dtype = Dtype.Default
        dsharp.config(dtype=Dtype.Float64)
        Assert.CheckEqual(Dtype.Float64, Dtype.Default)
        dsharp.config(dtype=dtype)

        // Restore the config before the test
        dsharp.config(configBefore)

    [<Test>]
    member _.TestBackends () =
        let backends = dsharp.backends() |> List.sort
        let backendsCorrect = [Backend.Reference; Backend.Torch] |> List.sort
        Assert.CheckEqual(backendsCorrect, backends)

    [<Test>]
    member _.TestDevices () =
        // Get devices for default reference backend
        let defaultReferenceBackendDevices = dsharp.devices()
        Assert.CheckEqual([Device.CPU], defaultReferenceBackendDevices)

        // Get devices for explicitly specified reference backend
        let explicitReferenceBackendDevices = dsharp.devices(backend=Backend.Reference)
        Assert.CheckEqual([Device.CPU], explicitReferenceBackendDevices)

        // Get CPU devices for explicitly specified reference backend
        let explicitReferenceBackendCPUDevices = dsharp.devices(backend=Backend.Reference, deviceType=DeviceType.CPU)
        Assert.CheckEqual([Device.CPU], explicitReferenceBackendCPUDevices)

        // Get devices for explicitly specified Torch backend
        let explicitTorchBackendDevices = dsharp.devices(backend=Backend.Torch)
        Assert.That(explicitTorchBackendDevices |> List.contains Device.CPU)
        let cudaAvailable = TorchSharp.torch.cuda.is_available()
        Assert.CheckEqual(cudaAvailable, (explicitTorchBackendDevices |> List.contains Device.GPU))

        let explicitTorchBackendDevices = dsharp.devices(backend=Backend.Torch)
        Assert.That(explicitTorchBackendDevices |> List.contains Device.CPU)
        let cudaAvailable = TorchSharp.torch.cuda.is_available()
        Assert.CheckEqual(cudaAvailable, (explicitTorchBackendDevices |> List.contains Device.GPU))

    [<Test>]
    member _.TestIsBackendAvailable () =
        let referenceBackendAvailable = dsharp.isBackendAvailable(Backend.Reference)
        Assert.That(referenceBackendAvailable)

    [<Test>]
    member _.TestIsDeviceAvailable () =
        let cpuAvailable = dsharp.isDeviceAvailable(Device.CPU)
        Assert.That(cpuAvailable)

    [<Test>]
    member _.TestIsCudaAvailable () =
        let cudaAvailable = dsharp.isCudaAvailable(Backend.Reference)
        Assert.False(cudaAvailable)

    [<Test>]
    member _.TestIsDeviceTypeAvailable () =
        Assert.That(dsharp.isDeviceTypeAvailable(DeviceType.CPU))

        Assert.That(dsharp.isDeviceTypeAvailable(DeviceType.CPU, Backend.Reference))
        Assert.False(dsharp.isDeviceTypeAvailable(DeviceType.CUDA, Backend.Reference))

        Assert.That(dsharp.isDeviceTypeAvailable(DeviceType.CPU, Backend.Torch))

        let cudaAvailable = TorchSharp.torch.cuda.is_available()
        let deviceSupported = dsharp.isDeviceTypeAvailable(DeviceType.CUDA, Backend.Torch)
        Assert.CheckEqual(cudaAvailable, deviceSupported)

    [<Test>]
    member _.TestTensorAPIStyles () =
        let x = dsharp.randn([5;5])

        // Base API
        dsharp.seed(0)
        let y1 = x.dropout(0.2).leakyRelu(0.1).sum(1)

        // PyTorch-like API
        dsharp.seed(0)
        let y2 = dsharp.sum(dsharp.leakyRelu(dsharp.dropout(x, 0.2), 0.1), 1)

        // Compositional API for pipelining Tensor -> Tensor functions (optional, accessed through TensorMath.Compose)
        dsharp.seed(0)
        let y3 = x |> dsharp.dropout 0.2 |> dsharp.leakyRelu 0.1 |> dsharp.sum 1

        Assert.CheckEqual(y1, y2)
        Assert.CheckEqual(y1, y3)

    [<Test>]
    member _.TestLoadSaveGeneric() =
        // string
        let v1 = "Hello, world!"
        let f1 = System.IO.Path.GetTempFileName()
        dsharp.save(v1, f1)
        let v1b = dsharp.load(f1)
        Assert.CheckEqual(v1, v1b)

        // int
        let v2 = 128
        let f2 = System.IO.Path.GetTempFileName()
        dsharp.save(v2, f2)
        let v2b = dsharp.load(f2)
        Assert.CheckEqual(v2, v2b)

        // float
        let v3 = 3.14
        let f3 = System.IO.Path.GetTempFileName()
        dsharp.save(v3, f3)
        let v3b = dsharp.load(f3)
        Assert.CheckEqual(v3, v3b)

        // bool
        let v4 = true
        let f4 = System.IO.Path.GetTempFileName()
        dsharp.save(v4, f4)
        let v4b = dsharp.load(f4)
        Assert.CheckEqual(v4, v4b)

        // list
        let v5 = [1, 2, 3]
        let f5 = System.IO.Path.GetTempFileName()
        dsharp.save(v5, f5)
        let v5b = dsharp.load(f5)
        Assert.CheckEqual(v5, v5b)

        // tuple
        let v6 = (1, 2, 3)
        let f6 = System.IO.Path.GetTempFileName()
        dsharp.save(v6, f6)
        let v6b = dsharp.load(f6)
        Assert.CheckEqual(v6, v6b)

        // dict
        let v7 = [("a", 1), ("b", 2), ("c", 3)]
        let f7 = System.IO.Path.GetTempFileName()
        dsharp.save(v7, f7)
        let v7b = dsharp.load(f7)
        Assert.CheckEqual(v7, v7b)

        // tuple of dicts
        let v8 = ([("a", 1), ("b", 2), ("c", 3)], [("a", 1), ("b", 2), ("c", 3)])
        let f8 = System.IO.Path.GetTempFileName()
        dsharp.save(v8, f8)
        let v8b = dsharp.load(f8)
        Assert.CheckEqual(v8, v8b)

        // tensor
        let v9 = dsharp.tensor([1, 2, 3])
        let f9 = System.IO.Path.GetTempFileName()
        dsharp.save(v9, f9)
        let v9b = dsharp.load(f9)
        Assert.CheckEqual(v9, v9b)

