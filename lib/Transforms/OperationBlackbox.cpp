#include "streamhls/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Dataflow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-operation-blackbox"

// Get the function name for a given arithmetic operation type
static std::string getBlackboxFunctionName(Operation *op) {
    std::string fn_name = "";
    if (isa<arith::AddFOp>(op)) fn_name += "addf";
    else if (isa<arith::MulFOp>(op)) fn_name += "mulf";
    else if (isa<arith::DivFOp>(op)) fn_name += "divf";
    else if (isa<arith::SubFOp>(op)) fn_name += "subf";
    else if (isa<math::ExpOp>(op)) fn_name += "exp_bb";
    else return "";
    if (!(op->getParentOfType<affine::AffineForOp>())) fn_name += "_ctrl_chain";
    return fn_name;
}

// Get or create a blackbox function declaration
static func::FuncOp getOrCreateBlackboxFunction(ModuleOp module, StringRef funcName, 
                                                 Type resultType, 
                                                 ArrayRef<Type> operandTypes) {
    // Check if function already exists
    if (auto existingFunc = module.lookupSymbol<func::FuncOp>(funcName)) {
        return existingFunc;
    }
    
    // Create function type
    FunctionType funcType = FunctionType::get(module.getContext(), operandTypes, resultType);
    
    // Create function declaration
    OpBuilder builder(module.getBody(), module.getBody()->end());
    auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();
    
    return func;
}

// Replace an operation with a blackbox function call
static void replaceOpWithBlackboxCall(Operation *op, ModuleOp module, 
                                           OpBuilder &builder) {
    std::string funcNameStr = getBlackboxFunctionName(op);
    if (funcNameStr.empty()) {
        return;
    }
    StringRef funcName(funcNameStr);
    
    // Get operand types and result type
    SmallVector<Type> operandTypes;
    for (Value operand : op->getOperands()) {
        operandTypes.push_back(operand.getType());
    }
    Type resultType = op->getResult(0).getType();
    
    // Get or create the blackbox function
    func::FuncOp blackboxFunc = getOrCreateBlackboxFunction(module, funcName, 
                                                           resultType, operandTypes);
    
    // Create function call before the operation
    builder.setInsertionPoint(op);
    auto callOp = builder.create<func::CallOp>(op->getLoc(), blackboxFunc, 
                                                op->getOperands());
    
    // Replace all uses of the operation result with the call result
    op->getResult(0).replaceAllUsesWith(callOp.getResult(0));
    
    // Erase the original operation
    op->erase();
}

namespace mlir {
namespace streamhls {

// Manually create all blackbox functions from MultiHeadSelfAttention.py (lines 5-43)
void createBlackboxFunctions(ModuleOp module) {
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);
    // Insert at the beginning of the module so functions are defined before use
    builder.setInsertionPointToStart(module.getBody());
    
    // Get f32 type
    Type f32Type = builder.getF32Type();
    Location loc = module.getLoc();
    
    // Function signatures:
    // - addf, mulf, subf, divf: (f32, f32) -> f32
    // - exp_bb: (f32) -> f32
    // Same for _ctrl_chain variants
    
    // Create function type for binary operations (2 args, 1 result)
    SmallVector<Type> binaryArgTypes = {f32Type, f32Type};
    FunctionType binaryFuncType = FunctionType::get(context, binaryArgTypes, f32Type);
    
    // Create function type for unary operations (1 arg, 1 result)
    SmallVector<Type> unaryArgTypes = {f32Type};
    FunctionType unaryFuncType = FunctionType::get(context, unaryArgTypes, f32Type);
    
    // Helper function to create a binary operation function
    auto createBinaryFunc = [&](StringRef name) {
        if (module.lookupSymbol<func::FuncOp>(name)) {
            LLVM_DEBUG(llvm::dbgs() << "Blackbox function " << name 
                      << " already exists, skipping creation\n");
            return;
        }
        
        // Create function with body
        auto func = builder.create<func::FuncOp>(loc, name, binaryFuncType);
        func.setPrivate();
        
        // Create entry block
        Block *entryBlock = func.addEntryBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(entryBlock);
        
        // Get function arguments
        Value arg0 = entryBlock->getArgument(0);
        Value arg1 = entryBlock->getArgument(1);
        
        // Determine operation type based on function name
        Value result;
        if (name == "addf" || name == "addf_ctrl_chain") {
            result = builder.create<arith::AddFOp>(loc, arg0, arg1);
        } else if (name == "mulf" || name == "mulf_ctrl_chain") {
            result = builder.create<arith::MulFOp>(loc, arg0, arg1);
        } else if (name == "subf" || name == "subf_ctrl_chain") {
            result = builder.create<arith::SubFOp>(loc, arg0, arg1);
        } else if (name == "divf" || name == "divf_ctrl_chain") {
            result = builder.create<arith::DivFOp>(loc, arg0, arg1);
        } else {
            llvm_unreachable("Unknown binary function name");
        }
        
        // Return the result (must be last operation in block)
        builder.create<func::ReturnOp>(loc, result);
        
        LLVM_DEBUG(llvm::dbgs() << "Created blackbox function with implementation: " << name << "\n");
    };
    
    // Helper function to create a unary operation function (exp_bb)
    auto createUnaryFunc = [&](StringRef name) {
        if (module.lookupSymbol<func::FuncOp>(name)) {
            LLVM_DEBUG(llvm::dbgs() << "Blackbox function " << name 
                      << " already exists, skipping creation\n");
            return;
        }
        
        // Create function with body
        auto func = builder.create<func::FuncOp>(loc, name, unaryFuncType);
        func.setPrivate();
        
        // Create entry block
        Block *entryBlock = func.addEntryBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(entryBlock);
        
        // Get function argument
        Value arg = entryBlock->getArgument(0);
        
        // Create constant 2.0
        FloatType f32FloatType = f32Type.cast<FloatType>();
        Value two = builder.create<arith::ConstantFloatOp>(
            loc, llvm::APFloat(2.0f), f32FloatType);
        
        // Multiply by 2 (placeholder for exp as per Python comment)
        Value result = builder.create<arith::MulFOp>(loc, arg, two);
        
        // Return the result (must be last operation in block)
        builder.create<func::ReturnOp>(loc, result);
        
        LLVM_DEBUG(llvm::dbgs() << "Created blackbox function with implementation: " << name << "\n");
    };
    
    // Create all binary functions
    createBinaryFunc("addf");
    createBinaryFunc("mulf");
    createBinaryFunc("subf");
    createBinaryFunc("divf");
    createBinaryFunc("addf_ctrl_chain");
    createBinaryFunc("mulf_ctrl_chain");
    createBinaryFunc("subf_ctrl_chain");
    createBinaryFunc("divf_ctrl_chain");
    
    // Create all unary functions
    createUnaryFunc("exp_bb");
    createUnaryFunc("exp_bb_ctrl_chain");
}

void insertBlackboxFunctionCalls(ModuleOp module, func::FuncOp func) {
    func.walk([&](func::CallOp op) {
        auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
        auto calleeFuncOp = dyn_cast<func::FuncOp>(callee);
        if (calleeFuncOp) {
            insertBlackboxFunctionCalls(module, calleeFuncOp);
        }
    });
    // Collect all operations to replace
    SmallVector<Operation *> opsToReplace;
    func.walk([&](Operation *op) {
        if (getBlackboxFunctionName(op) != "") {
            LLVM_DEBUG(llvm::dbgs() << "Replacing operation with blackbox function call: " << op->getName() << "\n");
            opsToReplace.push_back(op);
        }
    });
    
    // Replace each operation
    OpBuilder builder(func.getContext());
    for (Operation *op : opsToReplace) {
        replaceOpWithBlackboxCall(op, module, builder);
    }
}

} // namespace streamhls
} // namespace mlir

namespace {
    struct OperationBlackbox : public OperationBlackboxBase<OperationBlackbox> {
      void runOnOperation() override {
        LLVM_DEBUG(llvm::dbgs() << "Running operation blackbox pass\n");
        auto module = getOperation();
        
        // Create all blackbox functions upfront
        createBlackboxFunctions(module);
        
        for (auto func : module.getOps<func::FuncOp>()) {
            if (func.getName() == "forward") {
                insertBlackboxFunctionCalls(module, func);
            }
        }
    }
};
} // namespace

#define GEN_PASS_DEF_OPERATIONBLACKBOX
#include "streamhls/Transforms/Passes.h.inc"

std::unique_ptr<Pass> streamhls::createOperationBlackboxPass() {
  return std::make_unique<OperationBlackbox>();
}