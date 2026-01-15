first create the environment: source setup-env.sh

then build llvm: source build-llvm.sh

then build streamhls: source build-streamhls.sh

^^The above steps are handled in codesign full_env_start.sh script. For first build, you should put your ampl package directory into Stream-HLS, but can activate the license key after build is complete. Or, you can look at build-streamhls.sh and repeat the steps pertaining to ampl, then activate the license key.

from Stream-HLS directory: upload your ampl package to this directory (see readme)
and run: ampl
from the shell, run: shell "amplkey activate --uuid <license-uuid>";