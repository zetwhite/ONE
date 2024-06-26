# ifndef __ONERT_BACKEND_TRAIN_EXTRA_TENSOR_ALLOCATOR_H__
# define __ONERT_BACKEND_TRAIN_EXTRA_TENSOR_ALLOCATOR_H__

// ir region
#include "ir/train/TrainableOperationVisitor.h"

// backend regoin
#include "backend/train/TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace train
{

// Add KernelTensor(Tensor used only in *Layer) to KernelBuilder  
class ExtraTensorAllocator : public ir::train::TrainableOperationVisitor
{
public:

private:
  std::shared_ptr<TensorBuilder> _tensor_builder;
};

} // namespace train
} // namespace backend
} // namespace onert 


#endif __ONERT_BACKEND_TRAIN_EXTRA_TENSOR_ALLOCATOR_H__
