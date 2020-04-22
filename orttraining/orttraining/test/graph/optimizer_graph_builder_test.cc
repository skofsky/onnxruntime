// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder.h"
#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"
#include "orttraining/core/graph/adasum_optimizer_graph_builder.h"
#include "orttraining/core/graph/zero_optimizer_graph_builder.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/test_environment.h"

using onnxruntime::test::CountOpsInGraph;

namespace onnxruntime {
namespace training {
namespace test {
namespace {

const std::vector<const char*> k_weight_names{"W0", "W1", "W2", "W3", "W4"};
const std::vector<int64_t> k_weight_dims{160, 160, 80, 400, 80};
constexpr const char* const k_loss_scaling_factor_name = "loss_scaling_factor";
constexpr const char* const k_optimizer_op_name = "AdamOptimizer";
constexpr const char* const k_horovod_all_reduce_op_name = "HorovodAllReduce";
constexpr const char* const k_all_reduce_op_name = "NcclAllReduce";
constexpr const char* const k_all_gather_op_name = "NcclAllGather";
constexpr const char* const k_reduce_scatter_op_name = "NcclReduceScatter";
constexpr const char* const k_is_all_finite_op_name = "IsAllFinite";
constexpr const char* const k_gradient_norm_op_name = "ReduceAllL2";
constexpr const char* const k_unscale_op_name = "MixedPrecisionScale";
constexpr const char* const k_inplace_accumulator_op_name = "InPlaceAccumulator";
constexpr const char* const k_zero_gradient_op_name = "ZeroGradient";

class OptimizerGraphBuilderTest : public testing::Test {
 protected:
  OptimizerGraphBuilderTest() : model_{"test_model", false, onnxruntime::test::DefaultLoggingManager().DefaultLogger()},
                                graph_{model_.MainGraph()} {
  }

  virtual void SetUp() override {
    ASSERT_STATUS_OK(SetUpBaseGraph(graph_));
  }

  // sets up a base graph with weight and gradient NodeArgs for each weight name
  Status SetUpBaseGraph(Graph& graph) {
    std::unordered_set<std::string> weight_and_gradient_names{};
    for (size_t i = 0; i < k_weight_names.size(); i++) {
      std::string weight_name = k_weight_names[i];
      std::string gradient_name = GradientBuilderBase::GradientName(weight_name);
      int64_t dims = k_weight_dims[i];

      AddInitializer(graph, weight_name, dims);
      AddInitializer(graph, gradient_name, dims);

      weight_and_gradient_names.emplace(weight_name);
      weight_and_gradient_names.emplace(gradient_name);
    }

    Graph::ResolveOptions resolve_options{};
    resolve_options.initializer_names_to_preserve = &weight_and_gradient_names;
    return graph.Resolve(resolve_options);
  }

  void AddInitializer(Graph& graph, const std::string& name, int64_t dims) {
    ONNX_NAMESPACE::TypeProto float_tensor_type{};
    float_tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dims);
    graph.GetOrCreateNodeArg(name, &float_tensor_type);

    ONNX_NAMESPACE::TensorProto initializer{};
    initializer.set_name(name);
    initializer.add_dims(dims);
    initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t i = 0; i < dims; i++) {
      initializer.add_float_data(float(i));
    }
    graph.AddInitializedTensor(initializer);
  }

  Model model_;
  Graph& graph_;
};

std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap() {
  std::unordered_map<std::string, OptimizerNodeConfig> result{};
  std::transform(
      k_weight_names.begin(), k_weight_names.end(), std::inserter(result, result.end()),
      [](const std::string& weight_name) {
        return std::make_pair(
            weight_name, OptimizerNodeConfig{k_optimizer_op_name, nullptr, "Learning_Rate", {}});
      });
  return result;
}

OptimizerBuilderRegistry& GetOptimizerBuilderRegistry() {
  return OptimizerBuilderRegistry::GetInstance();
}

int GetOpCount(const std::map<std::string, int>& op_counts, const std::string& op_type) {
  auto op_count_it = op_counts.find(op_type);
  return op_count_it != op_counts.end() ? op_count_it->second : 0;
}

}  // namespace

static void TestDefaultOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  OptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap());

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());

  // verify distributed operations don't exist
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_reduce_scatter_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_all_gather_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_horovod_all_reduce_op_name), 0);
}

TEST_F(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

#if defined(USE_HOROVOD) || defined(USE_NCCL)
static void TestAllreduceOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  AllreduceOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap());

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify allreduce operations exist
  ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
  if (config.use_nccl) {
    ASSERT_GT(GetOpCount(op_counts, k_all_reduce_op_name), 0);
  } else {
    ASSERT_GT(GetOpCount(op_counts, k_horovod_all_reduce_op_name), 0);
  }

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}

static void TestAdasumOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  AdasumOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap());

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size() * 2);
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify allreduce operations exist
  ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
  ASSERT_GT(GetOpCount(op_counts, k_horovod_all_reduce_op_name), 0);

  // verify in place adder operations exist
  ASSERT_GT(GetOpCount(op_counts, k_inplace_accumulator_op_name), 0);

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}
#endif

#ifdef USE_HOROVOD
TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}
TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}
TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

#endif  // USE_HOROVOD

#ifdef USE_NCCL
TEST_F(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

static void TestZeROOptimizerGraphBuilder(OptimizerGraphConfig config, Model& model) {
  ZeROOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap());

  // Make a clone of the model/graph to test on.
  auto model_proto = model.ToProto();
  Model model_copy(model_proto, nullptr, onnxruntime::test::DefaultLoggingManager().DefaultLogger());
  Graph& graph = model_copy.MainGraph();

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify ZeRO operations exist
  ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
  ASSERT_GT(GetOpCount(op_counts, k_reduce_scatter_op_name), 0);
  ASSERT_GT(GetOpCount(op_counts, k_all_gather_op_name), 0);

  // verify optimizers exist
  ASSERT_GT(GetOpCount(op_counts, k_optimizer_op_name), 0);
  ASSERT_LT(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_NoMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 1;
    config.use_mixed_precision = false;
    TestZeROOptimizerGraphBuilder(config, model_);
  }
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_NoMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 10;
    config.use_mixed_precision = false;
    TestZeROOptimizerGraphBuilder(config, model_);
  }
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_WithMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 1;
    config.use_mixed_precision = true;
    config.loss_scale_input_name = k_loss_scaling_factor_name;
    TestZeROOptimizerGraphBuilder(config, model_);
  }
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 10;
    config.use_mixed_precision = true;
    config.loss_scale_input_name = k_loss_scaling_factor_name;
    TestZeROOptimizerGraphBuilder(config, model_);
  }
}

#endif  // USE_NCCL

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
