/*
 * Copyright (c) 2024
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#ifndef STREAMHLS_SUPPORT_TECHCONFIG_H
#define STREAMHLS_SUPPORT_TECHCONFIG_H

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace mlir {
namespace streamhls {

/// TechConfig holds the technology-specific latency and DSP usage values
/// for various arithmetic operations. It can be loaded from a JSON config file
/// or use default values if no config is provided.
class TechConfig {
public:
  /// Default constructor with hardcoded default values
  TechConfig();

  /// Load configuration from a JSON file. Returns true on success.
  /// If the file doesn't exist or can't be parsed, default values are used.
  bool loadFromFile(const std::string &configPath);

  /// Get the latency (in cycles) for an operation given its MLIR operation name
  /// e.g., "arith.addf", "arith.mulf", "arith.divf", "arith.subf", "math.exp"
  int64_t getLatency(llvm::StringRef opName) const;

  /// Get the DSP usage for an operation given its MLIR operation name
  int64_t getDspUsage(llvm::StringRef opName) const;

  /// Check if a config file was successfully loaded
  bool isConfigLoaded() const { return configLoaded; }

private:
  /// Initialize default values
  void initDefaults();

  /// Map from operation short name (fadd, fmul, etc.) to latency
  llvm::StringMap<int64_t> latencyMap;

  /// Map from operation short name to DSP usage
  llvm::StringMap<int64_t> dspUsageMap;

  /// Map from MLIR operation name to short name
  static llvm::StringRef getShortName(llvm::StringRef opName);

  bool configLoaded = false;
};

/// Global function to get the singleton TechConfig instance
/// This allows the config to be loaded once and used throughout the codebase
TechConfig &getTechConfig();

/// Initialize the global TechConfig from a file path
/// Should be called early in the pass pipeline if a config file is provided
bool initTechConfig(const std::string &configPath);

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_SUPPORT_TECHCONFIG_H
