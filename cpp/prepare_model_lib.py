import argparse
import json
import os
from tvm.contrib import utils, tar
from tvm.contrib import ndk, xcode


def main(args):
    path = args.tar_model_lib
    if not os.path.isfile(path):
        raise RuntimeError(f"Cannot find model library {path}")

    temp = utils.tempdir()
    objects = tar.normalize_file_list_by_unpacking_tars(temp, [path])
    if args.device == "android":
        ndk.create_shared("model_lib_android.so", objects)
    elif args.device == "metal":
        xcode.create_dylib("model_lib_metal.dylib", objects, arch="x86_64")
    else:
        raise RuntimeError(f"Cannot create library for {args.device}")
    #ndk.create_staticlib(os.path.join("build", "model_lib", "libmodel_android.a"), tar_list)
    #print(f"Creating lib from {tar_list}..")


if __name__ == "__main__":
    device_types = ["android", "metal", "linux"]
    parser = argparse.ArgumentParser(description='Prepare model lib compiled by SLM approach')
    parser.add_argument("--tar_model_lib", type=str, required=True, help="Tar archive with model")
    parser.add_argument(
        "--device",
        help=f"Select target device (Possible options: {device_types})",
        required=True,
        choices=device_types
    )
    args = parser.parse_args()
    if args.device not in device_types:
        raise RuntimeError(f"Incorrect device type {args.device}. You can choose one of the following {device_types}")
    main(args)

