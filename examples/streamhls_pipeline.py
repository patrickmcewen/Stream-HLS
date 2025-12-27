import argparse 
import os
import sys
import time
# path lib
from pathlib import Path
import logging
import subprocess


# while getopts "i:m:c:t:d:b:" flag; do
#     case "${flag}" in
#         i) prjs_path=${OPTARG};;
#         m) model=${OPTARG};;
#         c) dsps=${OPTARG};;  # Assuming 'c' is for 'dsps'
#         t) tilingLimit=${OPTARG};;  # Assuming 't' is for 'tilingLimit'
#         d) debugPoint=${OPTARG};;  # Assuming 'd' is for 'debugPoint'
#         b) combinedOptimization=${OPTARG};;  # Assuming 'b' is for 'combinedOptimization'
#         *) echo "Invalid option"; exit 1;;
#     esac
# done
# arg parse options
parser = argparse.ArgumentParser()
parser.add_argument('--prjsdir', type=str, required=False, default='tests')
parser.add_argument('--bench', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dsps', type=int, required=False, default=1024)
parser.add_argument('--tilelimit', type=int, required=False, default=16)
parser.add_argument('--debug', type=int, required=False, default=14)
parser.add_argument('--combOpt', type=int, required=False, default=0)
parser.add_argument('--permOpt', type=int, required=False, default=1)
parser.add_argument('--paralOpt', type=int, required=False, default=1)
parser.add_argument('--timelimit', type=int, required=False, default=10)
parser.add_argument('--bufferize', type=int, required=False, default=0)
parser.add_argument('--conv', type=int, required=False, default=0)
parser.add_argument('--minimize-on-chip-buffers', type=int, required=False, default=0)
parser.add_argument('--compile_only', type=int, required=False, default=0)


args = parser.parse_args()
print("prjsdir: ", args.prjsdir)
# prj_path=$(realpath $prjs_path/$model)
prjs_dir = args.prjsdir
model = args.model
bench = args.bench
prj_path = Path(prjs_dir) / model
# get the absolute path
#prj_path = prj_path.resolve()
dsps = args.dsps
tilelimit = args.tilelimit
debug = args.debug
combOpt = args.combOpt
permOpt = args.permOpt
paralOpt = args.paralOpt
timelimit = args.timelimit
bufferize = args.bufferize
conv = args.conv
minimize_on_chip_buffers = args.minimize_on_chip_buffers
compile_only = args.compile_only

config = {
  "Model": model,
  "Benchmark": bench,
  "DSPs Limit": dsps,
  "Tiling Limit": tilelimit,
  "Debug Point": debug,
  "Combined Optimization": combOpt,
  "Permutation Optimization": permOpt,
  "Parallelization Optimization": paralOpt,
  "Time Limit (mins)": timelimit
}
log_file = prj_path / 'streamhls'
log_file = f'{log_file}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'

if(combOpt == 1):
  permOpt = 1
  paralOpt = 1

combOpt = 'true' if combOpt == 1 else 'false'
permOpt = 'true' if permOpt == 1 else 'false'
paralOpt = 'true' if paralOpt == 1 else 'false'

# call torch mlir
if (compile_only == 0):
  print(bench)
  print(bench.split("/")[-1])
  cmd = f'python gen_mlir_designs.py -b codesign -m {model} -o {prjs_dir} --benchmark-path {bench}'
  os.system(cmd)
# exit(0)
# set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create file handler which logs even debug messages
# mkdir -p prj_path
prj_path.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    return result.returncode

if compile_only == 0:
  # start time
  start_time = time.time()

  # host pipeline
  cmd = f'streamhls-opt {prj_path}/mlir/input/{model}.mlir \
    -streamhls-host-pipeline \
    > {prj_path}/mlir/host/{model}.mlir'
  run_command(cmd)

  cmd = f'streamhls-translate {prj_path}/mlir/host/{model}.mlir \
    -emit-vivado-hls \
    -vitis-hls-weights-dir="{prj_path}/hls/data" \
    -vitis-hls-is-host=true \
    -o {prj_path}/hls/src/{model}_tb.cpp'
  run_command(cmd)

  logger.info(f'MLIR to HLS for {model}')

  # kernel pipeline
  cmd = f'streamhls-opt {prj_path}/mlir/input/{model}.mlir \
    -streamhls-kernel-pipeline="top-func=forward \
      graph-file={prj_path}/mlir/graphs/graph\
      report-file={prj_path}/mlir/intermediates/{model}\
      optimize-schedule={permOpt}\
      parallelize-nodes={paralOpt}\
      combined-optimization={combOpt}\
      board-dsps={dsps} \
      tiling-limit={tilelimit} \
      time-limit-minutes={timelimit} \
      bufferize-func-args={bufferize} \
      optimize-conv-reuse={conv} \
      minimize-on-chip-buffers={minimize_on_chip_buffers} \
      debug-point={debug}" \
    > {prj_path}/mlir/kernel/{model}.mlir'
  run_command(cmd)

  # copy prev command to prj_path/cmd.sh
  cmd_path = prj_path / 'cmd.sh'
  with open(cmd_path, 'w') as f:
    f.write(cmd)


  cmd = f'streamhls-translate {prj_path}/mlir/kernel/{model}.mlir \
    -emit-vivado-hls \
    -o {prj_path}/hls/src/{model}.cpp'
  run_command(cmd)

  end_time = time.time()
  # print(f'Time taken: {end_time - start_time} seconds')
  logger.info(f'Time taken: {end_time - start_time} seconds')

  # cpy hls.tcl to hls folder
  # python scripts/gen_vitis.py $model $prj_path/hls/run.sh
  # cp scripts/hls.tcl $prj_path/hls/hls.tcl
  # cmd = f'python scripts/gen_vitis.py {model} {prj_path}/hls/run.sh'
  cmd = f'python scripts/gen_batch.py {model} {prj_path}/hls/'
  run_command(cmd)
  cmd = f'cp scripts/hls.tcl {prj_path}/hls/hls.tcl'
  run_command(cmd)
  # print(f'Generated HLS for {model}')
  logger.info(f'Generated HLS for {model}')

# run csim
# print(f'Compiling {model}...')
logger.info(f'Compiling {model}...')
cmd = f'export PRJ_PATH={prj_path}/hls\n'
cmd += f'g++ {prj_path}/hls/src/{model}_tb.cpp {prj_path}/hls/src/{model}.cpp -lm -I${{XILINX_HLS}}/include -o {prj_path}/hls/{model}.bin\n'
cmd += f'cd {prj_path}/hls/ && ./{model}.bin'
run_command(cmd)
# print(f'Running csim for {model}...')
# print(f'Done!')
logger.info(f'Running csim for {model}...')
logger.info(f'Success!')

if compile_only == 0:
  report = {
    "Config": config,
    "Compilation Time (s)": end_time - start_time,
    "DSP Utilization": None,
    "Sequential Cycles": None,
    "Parallel Cycles": None,
    "Combined Cycles": None,
    "Permutation Design Space": None,
    "Parallelization Design Space": None,
    "Combined Design Space": None,
    "Permutation Search Time (s)": None,
    "Parallelization Search Time (s)": None,
    "Combined Search Time (s)": None,
    "Status": "Success"
  }
  # open log file and generate a report
  import json
  import re
  with open(log_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if('Permutation DesignSpaceSize:' in line):
        report["Permutation Design Space"] = int(re.findall(r'\d+', line)[0])
        break
    for line in lines:
      if('Parallelization DesignSpaceSize:' in line):
        report["Parallelization Design Space"] = int(re.findall(r'\d+', line)[0])
        break
    for line in lines:
      if('latency:' in line):
        report["Sequential Cycles"] = int(re.findall(r'\d+', line)[0])
        break
    for line in lines:
      if('Parallel Latency:' in line):
        report["Parallel Cycles"] = int(re.findall(r'\d+', line)[0])
        break
    for line in lines:
      if('Combined Latency:' in line):
        report["Combined Cycles"] = int(re.findall(r'\d+', line)[0])
        break
    for line in lines:
      if('Total DSPs:' in line):
        report["DSP Utilization"] = int(re.findall(r'\d+', line)[0])
        break
    for line in lines:
      if('Permutation solver: Pseudo-terminal will not be allocated because stdin is not a terminal.\n' in line):
        # get next line
        next_line = lines[lines.index(line) + 1]
        report["Permutation Search Time (s)"] = float(re.findall(r'\d+\.\d+', next_line)[0])
        break
    for line in lines:
      if('Parallelization solver: Pseudo-terminal will not be allocated because stdin is not a terminal.\n' in line):
        next_line = lines[lines.index(line) + 1]
        report["Parallelization Search Time (s)"] = float(re.findall(r'\d+\.\d+', next_line)[0])
        break
    for line in lines:
      if('Combined solver: Pseudo-terminal will not be allocated because stdin is not a terminal.\n' in line):
        next_line = lines[lines.index(line) + 1]
        report["Combined Search Time (s)"] = float(re.findall(r'\d+\.\d+', next_line)[0])
        break
    for line in lines:
      if('Error' in line):
        report["Status"] = "Failed"
        break

  if report["Sequential Cycles"] is None and report["Parallel Cycles"] is None and report["Combined Cycles"] is None:
    cmd = f'python {prj_path}/mlir/intermediates/{model}_default.py'
    output = subprocess.check_output(cmd, shell=True)
    report["Sequential Cycles"] = int(output)

  if report["Permutation Design Space"] is not None and report["Parallelization Design Space"] is not None:
    report["Combined Design Space"] = report["Permutation Design Space"] * report["Parallelization Design Space"]
  elif report["Permutation Design Space"] is not None:
    report["Combined Design Space"] = report["Permutation Design Space"]
  elif report["Parallelization Design Space"] is not None:
    report["Combined Design Space"] = report["Parallelization Design Space"]
  else:
    report["Combined Design Space"] = None
  # json report
  json_report = json.dumps(report, indent=2)
  # save to the same location as log file
  report_file = log_file.replace('.log', '.json')
  with open(report_file, 'w') as f:
    f.write(json_report)


