import argparse
import tempfile
import pickle
import os
import tvm
import time
from pathlib import Path
from tvm import te, tir
from typing import Any, Dict, Optional
from tvm.target import Target
from tvm import relax
from tvm.script import relax as R
from tvm.ir import IRModule
from tvm import tir
from tvm.relax.transform import LegalizeOps
from tvm import relay
from tvm.relay import testing
from tvm.relay.op.contrib import clml
from tvm import dlight as dl
import numpy as np
import tvm.relax.backend.contrib.cublas as _
from tvm.contrib.nvcc import parse_compute_version
from tvm.relax.backend import get_patterns_with_prefix
from tvm import meta_schedule as ms
import tvm.tir.tensor_intrin.cuda
from tvm.contrib import utils, ndk, xcode
from tvm import rpc
from tvm.relax.frontend.nn import Tensor, op
from tvm.meta_schedule.runner import RPCRunner, RPCConfig, EvaluatorConfig
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.runtime import Module, NDArray



from mlc_chat.support import logging
from tvm.relax.frontend import nn

logger = logging.getLogger(__name__)
dtype = "float16"

BATCH = 100
vocab_size = 32000
num_hidden_layers = 1
QUANTIZE = True
REDUCE_PIPELINE = True
USE_TIME_EVALUATOR = True
SPLIT_TO_SMALL_KERNELS = False
TUNE = True
USE_MS = True # Looks like only ms supported for Relax

if not REDUCE_PIPELINE:
    intermediate_size = 11008
else:
    intermediate_size = 12288
    parts = 6
    #intermediate_size = 10240
    #parts = 5
tune_work_dir = "./tune_{}_{}/".format(num_hidden_layers, intermediate_size)

@tvm.transform.module_pass(opt_level=0, name="DebugDump")
class _DebugDump:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging.
    Only enabled when debug_dump is not None"""

    def __init__(self, file_name: str, file_path: Optional[Path], show_meta: bool = False):
        self.file_name = file_name
        self.file_path = file_path
        self.show_meta = show_meta

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """A dummy transformation that dumps the module to file"""
        if self.file_path is not None:
            # NOTE: We use debug level here to avoid spamming the console
            #logger.debug("Dumping IR to %s", self.file_path / self.file_name)
            logger.debug("Dumping IR to %s", self.file_path + "/" + self.file_name)
            #with open(self.file_path / self.file_name, "w", encoding="utf-8") as f:
            with open(self.file_path + "/" + self.file_name, "w", encoding="utf-8") as f:
                f.write(mod.script(show_meta=self.show_meta))
        return mod


class LlamaFFN(nn.Module):
    def __init__(self, id):
        #tensor_parallel_shards = 1
        hidden_size = 4096
        str_id = str(id)
        super().__init__()
        self.intermediate_size = intermediate_size #// tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
            dtype=dtype,
            name_w="weights" + str_id
        )
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False, dtype=dtype, name_w="weights_1" + str_id)

    def forward(self, x: Tensor):
        print("x: ", x)
        if not REDUCE_PIPELINE:
            concat_x1_x2 = self.gate_up_proj(x)
            x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
            return self.down_proj(op.silu(x1) * x2)
        else:
            if SPLIT_TO_SMALL_KERNELS:
                splitted = op.split(x, 100, axis=1)
                arr = []
                for s in splitted:
                    arr.append(self.gate_up_proj(s))
                a = op.concat(arr, 1)
            else:
                a = self.gate_up_proj(x)
            x1, x2, x3, x4, x5, x6 = op.split(a, parts, axis=-1)
            return x1


class LlamaDecoderLayer(nn.Module):
    def __init__(self, id):
        self.mlp = LlamaFFN(id)


    def forward(self, hidden_states: Tensor):
        #out = self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        #hidden_states = self._apply_residual(out, residual=hidden_states)
        #out = self.mlp(self.post_attention_layernorm(hidden_states))
        out = self.mlp(hidden_states)
        return out


class LlamaModel(nn.Module):
    def __init__(self):
        hidden_size = 4096
        # rms_norm_eps = 1e-06
        self.embed_tokens = nn.Embedding("vocab_size", hidden_size, dtype=dtype, name="emb_params")
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(i) for i in range(num_hidden_layers)]
        )
        #self.norm = nn.RMSNorm(hidden_size, -1, rms_norm_eps, bias=False)

    def forward(self, input_embed: Tensor):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
        #hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCasualLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        hidden_size = 4096
        self.lm_head = nn.Linear(hidden_size, "vocab_size", bias=False, dtype=dtype, name_w='lm_head')
        self.model = LlamaModel()

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor):
        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed)
        if not REDUCE_PIPELINE:
            hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
            logits = self.lm_head(hidden_states)
            if logits.dtype != "float32":
                logits = logits.astype("float32")
        else:
            logits = hidden_states
        return logits


def get_pipeline(mod, target):
    from mlc_chat.compiler_pass.attach_to_ir_module import AttachMemoryPlanAttr
    from mlc_chat.compiler_pass.fuse_ft_dequantize_matmul_epilogue import FuseFTDequantizeEpilogue
    from mlc_chat.compiler_pass.fuse_dequantize_transpose import FuseDequantizeTranspose
    from mlc_chat.compiler_pass.fuse_add_norm import FuseAddRMSNorm
    from mlc_chat.compiler_pass.fuse_transpose_matmul import FuseTransposeMatmul
    from mlc_chat.compiler_pass.fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
    from mlc_chat.compiler_pass.fuse_dequantize_take import FuseDequantizeTake
    from mlc_chat.compiler_pass.clean_up_tir_attrs import CleanUpTIRAttrs
    from mlc_chat.compiler_pass.lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc
    from mlc_chat.compiler_pass.scatter_tuple_get_item import ScatterTupleGetItem
    debug_dump = './tmp/'
    seq = tvm.transform.Sequential(
        [
            # Phase 0. Add additional information for compilation and remove unused Relax func
            AttachMemoryPlanAttr(),
            tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False)),
            # Phase 1. Passes on high-level operator graph
            FuseFTDequantizeEpilogue(),
            FuseDequantizeTranspose(),
            FuseAddRMSNorm(target=target),
            FuseTransposeMatmul(),
            # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.AnnotateTIROpPattern(),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.FuseOps(),
            tvm.relax.transform.FuseTIR(),
            # Phase 3. Passes on TIR
            FuseDequantizeMatmulEwise(),
            FuseDequantizeTake(),
            tvm.relax.transform.DeadCodeElimination(),
            CleanUpTIRAttrs(["op_pattern"]),
            ### Phase 4. Low-level Optimizations
            ##dl.ApplyDefaultSchedule(
            ##    dl.gpu.Matmul(),
            ##    dl.gpu.GEMV(),
            ##    dl.gpu.Reduction(),
            ##    dl.gpu.GeneralReduction(),
            ##    dl.gpu.Fallback(),
            ##),
            ##_DebugDump("after_dlight.py", debug_dump, show_meta=False),
            LiftTIRGlobalBufferAlloc(),
            (
                tvm.tir.transform.ForceNarrowIndexToInt32()
                if target.kind.name != "cuda"
                else tvm.transform.Sequential([])
            ),
            ScatterTupleGetItem(),
            tvm.relax.transform.RewriteDataflowReshape(),
            tvm.relax.transform.ToNonDataflow(),
            tvm.relax.transform.RemovePurityChecking(),
            tvm.relax.transform.CallTIRRewrite(),
            tvm.relax.transform.StaticPlanBlockMemory(),
            ###AttachMetadataWithMemoryUsage(metadata),
            tvm.relax.transform.RewriteCUDAGraph(),
            tvm.relax.transform.LowerAllocTensor(),
            tvm.relax.transform.KillAfterLastUse(),
            tvm.relax.transform.VMBuiltinLower(),
            tvm.relax.transform.VMShapeLower(),
            tvm.relax.transform.AttachGlobalSymbol(),
            _DebugDump("final.py", debug_dump, show_meta=False),
        ]
    )
    mod = seq(mod)
    return mod

def get_model_v2(target: Target):
    from mlc_chat.loader import QuantizeMapping
    from mlc_chat.quantization import AWQQuantize, FTQuantize, GroupQuantize, NoQuantize

    quantization = GroupQuantize(name='q4f16_1', kind='group-quant', group_size=32, quantize_dtype='int4', storage_dtype='uint32', model_dtype='float16', linear_weight_layout='NK')
    rms_norm_eps = 1e-06
    hidden_size = 4096
    bb = relax.BlockBuilder()
    with bb.function("prefill"):
        casual = LlamaForCasualLM()
        if QUANTIZE:
            casual.to(quantization.model_dtype)
            quant_map = QuantizeMapping({}, {})
            casual = quantization.quantize_model(
                casual,
                quant_map,
                "",
            )
        with bb.dataflow():
            if not REDUCE_PIPELINE:
                input = Tensor.placeholder((1, "seq_len"), dtype="int32", name="A")
                params = [input._expr]
                input = casual.embed(input)
            else:
                input = Tensor.placeholder((1, 100, hidden_size), dtype=dtype, name="A")
                params = [input._expr]
            res = casual.prefill(input)
            params += [i._expr for i in casual.parameters()]
            gv = bb.emit_output(res._expr)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("prefill")
    bb.update_func(gv, mod[gv].with_attr("prefill", 1))
    print("+" * 10)
    print(mod)
    print("+" * 10)

    with target:
        mod = get_pipeline(mod, target)
    print("-" * 10)
    print(mod)
    print("-" * 10)

    return mod, params


def connect_rpc(rpc_host="127.0.0.1", rpc_port=9190, rpc_key="android"):
    tracker = rpc.connect_tracker(rpc_host, rpc_port)
    return tracker.request(rpc_key, priority=0)


if __name__ == "__main__":
    android = True
    fcompile_args = {}
    if android is True:
        target_c = "opencl"
        target_h = "llvm -mtriple=arm64-linux-android"
    else:
        target_c = "metal"
        target_h = "llvm"
    rpc_host = "127.0.0.1"
    rpc_port = 9190
    rpc_key = "android"

    target = Target(target_c, host=target_h)
    relax_lib = "relax_lib.so"
    initializer = relay.testing.init.Xavier()
    np.random.seed(0)

    print("> Before get_model")
    relax_mod, params = get_model_v2(target)
    if TUNE:
        if USE_MS:

            rpc_config = RPCConfig(
                tracker_host=rpc_host,
                tracker_port=rpc_port,
                tracker_key=rpc_key,
                session_priority=1,
                session_timeout_sec=100,
            )
            evaluator_config = EvaluatorConfig(
                number=3,
                repeat=1,
                min_repeat_ms=0,
            )
            runner = RPCRunner(rpc_config, evaluator_config)
            ms.tir_integration.tune_tir(
                mod=relax_mod,
                target=target,
                work_dir=tune_work_dir,
                #max_trials_global=100500,
                #max_trials_per_task=100500,
                max_trials_global=4096,
                max_trials_per_task=4096,
                #max_trials_per_task=256,
                num_trials_per_iter=32,
                #cost_model="random",
                #strategy="iterate-all",
                builder=LocalBuilder(
                    f_export="meta_schedule.builder.export_ndk",
                ),
                runner=runner,
                #space=dtune_space_gen()
            )
        else:
            from tvm import relay, auto_scheduler
            params = {}
            mod = tvm.IRModule({"main": relax_mod["prefill"]})
            tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)
            print(tasks)
        exit(0)

    if android is True:
        remote = connect_rpc(rpc_host, rpc_port, rpc_key)
        dev = remote.cl(0)
        fcompile = ndk.create_shared
    else:
        fcompile_args["sdk"] = "macosx"
        fcompile_args["arch"] = "x86_64"
        remote = rpc.LocalSession()
        dev = remote.metal(0)
        fcompile = xcode.create_dylib
    print("> Before build and export")
    relax.build(relax_mod, target).export_library(relax_lib, fcompile=fcompile, **fcompile_args)
    print("> Before upload")
    remote.upload(relax_lib)
    print("> Before load_module")
    rlib = remote.load_module(relax_lib)
    print("> Before VM create")
    vm = relax.VirtualMachine(rlib, dev)

    K = 4096
    if not REDUCE_PIPELINE:
        a_shape = (1, BATCH)
        input_dtype = "int32"
    else:
        a_shape = (1, BATCH, K)
        input_dtype = dtype
    print("> Before data initialization")
    M = intermediate_size * 2
    param_shape = (M, K)
    param_shape_q = (M, 512)
    param_shape_s = (M, 128)
    param_1_shape = (K, intermediate_size)
    param_1_shape_q = (K, intermediate_size // 8)
    param_1_shape_s = (K, intermediate_size // 32)
    lm_head_shape = (vocab_size, K)
    lm_head_shape_q = (vocab_size, 512)
    lm_head_shape_s = (vocab_size, 128)
    emb_params_shape = (vocab_size, K)
    emb_params_shape_q = (vocab_size, 512)
    emb_params_shape_s = (vocab_size, 128)

    a_data = np.zeros(a_shape).astype(input_dtype)
    lm_head_data = np.ones(lm_head_shape).astype(dtype)
    lm_head_data_q = np.ones(lm_head_shape_q).astype("uint32")
    lm_head_data_s = np.ones(lm_head_shape_s).astype(dtype)
    emb_params_data = np.ones(emb_params_shape).astype(dtype)
    emb_params_data_q = np.ones(emb_params_shape_q).astype("uint32")
    emb_params_data_s = np.ones(emb_params_shape_s).astype(dtype)
    initializer("weight", a_data)
    initializer("weight", emb_params_data_s)
    if QUANTIZE:
        data_np = {
            'A': a_data,
            'lm_head_q': lm_head_data_q,
            'lm_head_s': lm_head_data_s,
            'emb_params_q': emb_params_data_q,
            'emb_params_s': emb_params_data_s,
        }
    else:
        data_np = {
            'A': a_data,
            'lm_head': lm_head_data,
            'emb_params': emb_params_data,
        }

    for i in range(num_hidden_layers):
        str_i = str(i)
        param_data = np.zeros(param_shape).astype(dtype)
        param_data_q = np.zeros(param_shape_q).astype("uint32")
        param_data_s = np.zeros(param_shape_s).astype(dtype)
        param_1_data = np.zeros(param_1_shape).astype(dtype)
        param_1_data_q = np.zeros(param_1_shape_q).astype("uint32")
        param_1_data_s = np.zeros(param_1_shape_s).astype(dtype)
        initializer("weight", param_data_s)
        initializer("weight", param_1_data_s)
        if QUANTIZE:
            data_np['weights' + str_i + '_q'] = param_data_q
            data_np['weights' + str_i + '_s'] = param_data_s
            data_np['weights_1' + str_i + '_q'] = param_1_data_q
            data_np['weights_1' + str_i + '_s'] = param_1_data_s
        else:
            data_np['weights' + str_i] = param_data
            data_np['weights_1' + str_i] = param_1_data

    print("> Before data to TVM conversion")
    data = {}
    for k, v in data_np.items():
        data[k] = tvm.nd.array(v, dev)

    print("> Before set_input")
    vm.set_input("prefill", **data)
    for i in range(10):
        print("before run ", 10 - i)
        time.sleep(1)
    if USE_TIME_EVALUATOR:
        score_s = vm.time_evaluator("invoke_stateful", dev=dev, number=3, repeat=1)("prefill")
        print(score_s)
    else:
        vm.invoke_stateful("prefill")

        time.sleep(5)
        vm.invoke_stateful("prefill")



