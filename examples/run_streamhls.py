from data import model_configs
import os
import sys
import argparse 

parser = argparse.ArgumentParser(description='Run a single experiment')
parser.add_argument('-b', '--benchmark', type=str, help='Benchmark to run')
parser.add_argument('-k', '--kernel', type=str, help='Kernel to run')
parser.add_argument('-O', '--opt', type=int, default=0, help='Optimization level')
parser.add_argument('-c', '--compile_only', type=int, default=0, help='Compile only')
parser.add_argument('-d', '--outdir', type=str, default='designs', help='Output directory')
parser.add_argument('--dsps', type=int, default=2560*3, help='Number of DSPs')
parser.add_argument('--tilelimit', type=int, default=10, help='Tile limit')
parser.add_argument('--timelimit', type=int, default=20, help='Time limit')
parser.add_argument('--bufferize', type=int, default=0, help='Bufferize function arguments')
parser.add_argument('--conv', type=int, default=1, help='Enable conv optimization in StreamHLS pipeline')
parser.add_argument('--tech-config', type=str, default='', help='Path to technology config JSON file')
parser.add_argument('--codesign-mode', type=str, default='', choices=['', 'emit', 'apply'], help="Use the codesign pipeline: 'emit' writes the current design point, 'apply' applies a design point from JSON")
parser.add_argument('--solution-file', type=str, default='', help='Design-point JSON consumed in codesign apply mode')
parser.add_argument('--dump-pass-ir', action='store_true', help='Dump MLIR after each StreamHLS kernel/codesign pass to a separate log file')
parser.add_argument('--dump-pass-ir-diffs', action='store_true', help='With --dump-pass-ir, append a unified diff against the previous dump after each IR dump')
parser.add_argument('--pass-ir-log', type=str, default='', help='Path for --dump-pass-ir output; defaults under the design mlir/intermediates directory')
args = parser.parse_args()


tilelimit=args.tilelimit
timelimit=args.timelimit
dsps=args.dsps
conv=args.conv
tech_config=args.tech_config
# bufferize function arguments flag
bufferize=args.bufferize
# minimize_on_chip_buffers function arguments flag
minimize_on_chip_buffers=0
dbg_point=14
benchmark=args.benchmark
kernel = args.kernel
compile_only = args.compile_only
opt = args.opt

if opt == 5:
  permOpt=1
  paralOpt=1
  combOpt=1
elif opt == 4:
  permOpt=1
  paralOpt=1
  combOpt=0
elif opt == 3:
  permOpt=0
  paralOpt=1
  combOpt=0
elif opt == 2:
  permOpt=1
  paralOpt=0
  combOpt=0
elif opt == 1:
  permOpt=0
  paralOpt=0  
  combOpt=0


outDir=f'designs/{benchmark}/opt{opt}/{kernel}_{dsps}' if args.outdir == "designs" else args.outdir

print(f"outDir: {outDir}")

tech_config_arg = f'--tech-config={tech_config}' if tech_config else ''
codesign_arg = f'--codesign-mode={args.codesign_mode}' if args.codesign_mode else ''
solution_arg = f'--solution-file={args.solution_file}' if args.solution_file else ''
dump_pass_ir_arg = '--dump-pass-ir' if args.dump_pass_ir else ''
dump_pass_ir_diffs_arg = '--dump-pass-ir-diffs' if args.dump_pass_ir_diffs else ''
pass_ir_log_arg = f'--pass-ir-log={args.pass_ir_log}' if args.pass_ir_log else ''
cmd = f'python streamhls_pipeline.py \
  --prjsdir={outDir} \
  --bench={benchmark} \
  --model={kernel} \
  --permOpt={permOpt} \
  --paralOpt={paralOpt} \
  --combOpt={combOpt} \
  --dsps={dsps} \
  --tilelimit={tilelimit} \
  --timelimit={timelimit} \
  --bufferize={bufferize}\
  --minimize-on-chip-buffers={minimize_on_chip_buffers}\
  --debug={dbg_point}\
  --compile_only={compile_only} \
  --conv={conv} \
  {tech_config_arg} \
  {codesign_arg} \
  {solution_arg} \
  {dump_pass_ir_arg} \
  {dump_pass_ir_diffs_arg} \
  {pass_ir_log_arg}'
os.system(cmd)
print(f'Finished {kernel}...')
