/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//
/*
 * Modified by Suhail Basalama in 2024.
 *
 * This software is also released under the MIT License:
 * https://opensource.org/licenses/MIT
 */
#include "streamhls/Translation/EmitVivadoHLS.h"
#include "streamhls/Translation/EmitTAPA.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllTranslations();
  mlir::streamhls::registerEmitVivadoHLSTranslation();
  mlir::streamhls::registerEmitTAPATranslation();

  return failed(mlir::mlirTranslateMain(
      argc, argv, "STREAMHLS MLIR Dialect Translation Tool", registry));
}
