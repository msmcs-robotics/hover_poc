import os

GENERATED_DIR = "src/generated"

Import("env")

env.Execute("$PYTHONEXE -m pip install nanopb")

if not os.path.exists(GENERATED_DIR):
    os.mkdir(GENERATED_DIR)

env.Execute("nanopb_generator -D src/generated --strip-path -I ./proto ./proto/channels.proto")