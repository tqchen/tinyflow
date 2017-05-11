# TinyFlow: Build Your Own DL System in 2K Lines

TinyFlow is  "example code" for [NNVM](https://github.com/dmlc/nnvm/).

It demonstrates how can we build a clean, minimum and powerful computational
graph based deep learning system with same API as TensorFlow.
The operator code are implemented with [Torch7](https://github.com/torch/torch7) to reduce the effort to write operators while still demonstrating the concepts of the system (and Embedding Lua in C++ is kinda of fun:).

TinyFlow is a real deep learning system that can run on GPU and CPUs.
To support the examples, it takes.
- 927 lines code for operators
- 734 lines of code for execution runtime
- 71 lines of code for API glue
- 233 lines of code for front-end

Note that more code in operators can easily be added to make it as
feature complete as most existing deep learning systems.



## What is it for
As explained in the goal of [NNVM](https://github.com/dmlc/nnvm/),
it is important to make modular and reusable components for to enable us to build
customized learning system easily.

- Course Material for teaching DL system. TinyFlow can be used to teach student the concepts in deep learning systems.
  - e.g. design homeworks on implementing symbolic differentiation, memory allocation, operator fusion.
- Experiment bed for learning system researchers. TinyFlow allows easy addition with new system features with
  the modular design being portable to other system that reuses NNVM.
- Showcase of intermediate representation usecase. It demonstrates how intermediate representation like NNVM to be able to
  target multiple front-ends(TF, MXNet) and backends(Torch7, MXNet) with common set of optimizations.
- Test bed on common reusable modules for DL system. TinyFlow, together with other systems(e.g. MXNet) can be used as testbed on the
  common reusable modules in deep learning to encourage front-end, optimization module and backends
  that are shared across frameworks.
- Just for fun :)


We believe the Unix Philosophy can building learning system more fun and everyone can be able to build
and understand learning system better.

## Deep Learning System Course
If you are interested in learning how to build deep learning system from scratch, checkout [CSE 599G1: Deep Learning System](http://dlsys.cs.washington.edu/) from University of Washington. 

## The Design
- The graph construction API is automatically reused from NNVM
- We choose Torch7 as the default operator execution backend.
  - So TinyFlow can also be called "TorchFlow" since it is literally TensorFlow on top of Torch:)
  - This allows us to quickly implement the operators and focus code on the system part.
- We intentionally choose to avoid using [MXNet](https://github.com/dmlc/mxnet) as front or backend,
  since MXNet already uses NNVM as intermediate layer, and it would be more fun to try something different.

Although it is minimum. TinyFlow still comes with many advanced design concepts in Deep Learning system.
- Automatic differentiation.
- Shape/type inference.
- Static memory allocation for graph for memory efficient training/inference.

The operator implementation is easy Thanks to Torch7. More fun demonstrations will be added to the project.

## Dependencies
Most of TinyFlow's code is self-contained.
- TinyFlow depend on Torch7 for operator supports with minimum code.
  - We use a lightweight lua bridge code from dmlc-core/dmlc/lua.h
- NNVM is used for graph representation and optimizations

## Build
- Install Torch7
  - For OSX User, please install Torch with Lua 5.1 instead of LuaJIT,
    i.e. ```TORCH_LUA_VERSION=LUA51 ./install.sh```
- Set up environment variable ```TORCH_HOME``` to root of torch
- Type ```make```
- Setup python path to include tinyflow and nnvm
```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/tinyflow/python:/path/to/tinyflow/nnvm/python
```
- Try example program ```python example/mnist_softmax.py```

## Enable Fusion in TinyFlow
- Build NNVM with Fusion: uncomment fusion plugin part in config.mk, then `make`
- Build TinyFlow: enable `USE_FUSION` in Makefile, then `make`
- Try Example program `example/mnist_lenet.py`, change the config of session from `tf.Session(config='gpu')` to `tf.Session(config='gpu fusion')`
