/*
 * Copyright (c) 2024
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "streamhls/Support/TechConfig.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "streamhls-techconfig"

namespace mlir {
namespace streamhls {

// Static global instance
static TechConfig globalTechConfig;

TechConfig::TechConfig() {
  initDefaults();
}

void TechConfig::initDefaults() {
  // Default latency values (in cycles)
  // These are based on typical FPGA implementations
  latencyMap["fadd"] = 4;   // Floating-point addition
  latencyMap["fsub"] = 4;   // Floating-point subtraction
  latencyMap["fmul"] = 4;   // Floating-point multiplication
  latencyMap["fdiv"] = 15;  // Floating-point division
  latencyMap["fcmp"] = 1;   // Floating-point comparison
  latencyMap["fexp"] = 8;   // Exponential function

  // Default DSP usage values
  dspUsageMap["fadd"] = 2;  // Floating-point addition
  dspUsageMap["fsub"] = 2;  // Floating-point subtraction
  dspUsageMap["fmul"] = 3;  // Floating-point multiplication
  dspUsageMap["fdiv"] = 0;  // Division typically uses LUTs, not DSPs
  dspUsageMap["fcmp"] = 0;  // Comparison uses LUTs
  dspUsageMap["fexp"] = 7;  // Exponential uses multiple DSPs
}

bool TechConfig::loadFromFile(const std::string &configPath) {
  if (configPath.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No config file specified, using defaults\n");
    return false;
  }

  // Read the config file
  std::string errorMessage;
  auto configFile = mlir::openInputFile(configPath, &errorMessage);
  if (!configFile) {
    llvm::errs() << "Warning: Could not open tech config file '" << configPath
                 << "': " << errorMessage << "\n";
    llvm::errs() << "Using default latency and DSP values.\n";
    return false;
  }

  // Parse JSON
  auto config = llvm::json::parse(configFile->getBuffer());
  if (!config) {
    llvm::errs() << "Warning: Failed to parse JSON in tech config file '"
                 << configPath << "'\n";
    llvm::errs() << "Using default latency and DSP values.\n";
    return false;
  }

  auto configObj = config.get().getAsObject();
  if (!configObj) {
    llvm::errs() << "Warning: Expected JSON object in tech config file\n";
    llvm::errs() << "Using default latency and DSP values.\n";
    return false;
  }

  // Load latency values from "latency" section
  if (auto latencyObj = configObj->getObject("latency")) {
    if (auto val = latencyObj->getInteger("fadd"))
      latencyMap["fadd"] = *val;
    if (auto val = latencyObj->getInteger("fsub"))
      latencyMap["fsub"] = *val;
    if (auto val = latencyObj->getInteger("fmul"))
      latencyMap["fmul"] = *val;
    if (auto val = latencyObj->getInteger("fdiv"))
      latencyMap["fdiv"] = *val;
    if (auto val = latencyObj->getInteger("fcmp"))
      latencyMap["fcmp"] = *val;
    if (auto val = latencyObj->getInteger("fexp"))
      latencyMap["fexp"] = *val;
  }

  // Load DSP usage values from "dsp_usage" section
  if (auto dspObj = configObj->getObject("dsp_usage")) {
    if (auto val = dspObj->getInteger("fadd"))
      dspUsageMap["fadd"] = *val;
    if (auto val = dspObj->getInteger("fsub"))
      dspUsageMap["fsub"] = *val;
    if (auto val = dspObj->getInteger("fmul"))
      dspUsageMap["fmul"] = *val;
    if (auto val = dspObj->getInteger("fdiv"))
      dspUsageMap["fdiv"] = *val;
    if (auto val = dspObj->getInteger("fcmp"))
      dspUsageMap["fcmp"] = *val;
    if (auto val = dspObj->getInteger("fexp"))
      dspUsageMap["fexp"] = *val;
  }

  configLoaded = true;
  LLVM_DEBUG(llvm::dbgs() << "Loaded tech config from '" << configPath << "'\n");
  LLVM_DEBUG(llvm::dbgs() << "  fadd latency: " << latencyMap["fadd"] << ", DSP: " << dspUsageMap["fadd"] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  fsub latency: " << latencyMap["fsub"] << ", DSP: " << dspUsageMap["fsub"] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  fmul latency: " << latencyMap["fmul"] << ", DSP: " << dspUsageMap["fmul"] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  fdiv latency: " << latencyMap["fdiv"] << ", DSP: " << dspUsageMap["fdiv"] << "\n");

  return true;
}

llvm::StringRef TechConfig::getShortName(llvm::StringRef opName) {
  // Map MLIR operation names to short config names
  if (opName == "arith.addf")
    return "fadd";
  if (opName == "arith.subf")
    return "fsub";
  if (opName == "arith.mulf")
    return "fmul";
  if (opName == "arith.divf")
    return "fdiv";
  if (opName == "arith.cmpf")
    return "fcmp";
  if (opName == "math.exp")
    return "fexp";
  // Default: return empty, caller should use default value
  return "";
}

int64_t TechConfig::getLatency(llvm::StringRef opName) const {
  auto shortName = getShortName(opName);
  if (shortName.empty()) {
    // Default latency for unknown operations
    return 2;
  }
  auto it = latencyMap.find(shortName);
  if (it != latencyMap.end()) {
    return it->second;
  }
  // Default latency if not found
  return 2;
}

int64_t TechConfig::getDspUsage(llvm::StringRef opName) const {
  auto shortName = getShortName(opName);
  if (shortName.empty()) {
    // Default DSP usage for unknown operations
    return 0;
  }
  auto it = dspUsageMap.find(shortName);
  if (it != dspUsageMap.end()) {
    return it->second;
  }
  // Default DSP usage if not found
  return 0;
}

TechConfig &getTechConfig() {
  return globalTechConfig;
}

bool initTechConfig(const std::string &configPath) {
  return globalTechConfig.loadFromFile(configPath);
}

} // namespace streamhls
} // namespace mlir
