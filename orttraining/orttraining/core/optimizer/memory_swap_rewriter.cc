// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/graph/graph_utils.h"
#include "memory_swap_rewriter.h"

namespace onnxruntime {

static bool IsBackwardNode(const Node& node) {
  return node.Description() == "Backward pass";
}

static void ComputeTopoIndices(const Graph& graph, std::unordered_map<NodeIndex, int>& topo_indices) {
  GraphViewer graph_viewer(graph);
  int topo_index = 0;
  topo_indices.clear();
  for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    topo_indices.insert(std::make_pair(index, topo_index++));
  }
}

static bool ShouldHandleSrcNode(const Node& node) {
  // blacklist some ops for memory swap
  // TODO: make it configurable
  static const std::unordered_set<std::string> ignore_src_op_types =
      {"Shape",
       "Reshape",
       "Transpose",
       "Cast"};
  return !IsBackwardNode(node) && 0 == ignore_src_op_types.count(node.OpType());
}

static bool ShouldHandleDstNode(const Node& node, int dst_arg_idx) {
  // whitelist ops and arg_idx for memory swap
  static const std::unordered_map<std::string, std::unordered_set<int>> allowed_dst_op_args =
      {{"Reshape", {0}},
       {"Gemm", {0, 1}}};
  return IsBackwardNode(node) &&
         allowed_dst_op_args.count(node.OpType()) &&
         allowed_dst_op_args.at(node.OpType()).count(dst_arg_idx);
}

Status MemorySwapRewriter::Apply(Graph& graph, Node& src_node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  std::unordered_set<int> to_bw_arg_idx;
  for (auto edge_iter = src_node.OutputEdgesBegin(); edge_iter != src_node.OutputEdgesEnd(); ++edge_iter) {
    if (ShouldHandleDstNode(edge_iter->GetNode(), edge_iter->GetDstArgIndex())) {
      to_bw_arg_idx.insert(edge_iter->GetSrcArgIndex());
    }
  }
  for (int src_node_output_idx : to_bw_arg_idx) {
    NodeArg* src_node_output_arg = const_cast<NodeArg*>(src_node.OutputDefs()[src_node_output_idx]);
    auto& swap_out_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_out", src_node_output_arg->TypeAsProto());
    auto& swap_in_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_in", src_node_output_arg->TypeAsProto());
    auto& swap_out_node = graph.AddNode(src_node_output_arg->Name() + "_swapout",
                                        "SwapToHost",
                                        "",
                                        {src_node_output_arg},
                                        {&swap_out_arg});
    auto& swap_in_node = graph.AddNode(src_node_output_arg->Name() + "_swapin",
                                       "SwapFromHost",
                                       "Backward pass",
                                       {&swap_out_arg},
                                       {&swap_in_arg});

    // process output edges from this output_def
    // note this needs to happen before linking src_node with swap_out_node
    // and since the operation might change src_node's OutputEdges, needs a copy of original edges
    Node::EdgeSet src_node_output_edges(src_node.OutputEdgesBegin(), src_node.OutputEdgesEnd());
    for (const auto& output_edge : src_node_output_edges) {
      if (output_edge.GetSrcArgIndex() != src_node_output_idx)
        continue;

      if (!ShouldHandleDstNode(output_edge.GetNode(), output_edge.GetDstArgIndex()))
        continue;

      const Node& dst_node = output_edge.GetNode();
      int dst_arg_idx = output_edge.GetDstArgIndex();
      // remove edge from src_node to dst_node
      graph.RemoveEdge(src_node.Index(), dst_node.Index(), src_node_output_idx, dst_arg_idx);
      // add edge from swap_in to dst_node
      graph.AddEdge(swap_in_node.Index(), dst_node.Index(), 0, dst_arg_idx);
    }

    // add edges in graph
    graph.AddEdge(src_node.Index(), swap_out_node.Index(), src_node_output_idx, 0);
    graph.AddEdge(swap_out_node.Index(), swap_in_node.Index(), 0, 0);
  }
  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

bool MemorySwapRewriter::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  // only check forward nodes
  if (!ShouldHandleSrcNode(node))
    return false;

  static const Graph* last_graph = nullptr;
  static std::unordered_map<NodeIndex, int> topo_indices;
  if (last_graph != &graph) {
    last_graph = &graph;
    ComputeTopoIndices(graph, topo_indices);
  }

  // check if the node has one output going to a backward
  int fw_topo_idx = topo_indices[node.Index()];
  for (auto iter = node.OutputEdgesBegin(); iter != node.OutputEdgesEnd(); ++iter) {
    if (ShouldHandleDstNode(iter->GetNode(), iter->GetDstArgIndex())) {
      int bw_topo_idx = topo_indices[iter->GetNode().Index()];
      if (bw_topo_idx - fw_topo_idx > min_topo_distance_)
        return true;
    }
  }
  return false;
}

Status AddControlEdgeForMemorySwapRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  std::unordered_map<NodeIndex, int> topo_indices;
  ComputeTopoIndices(graph, topo_indices);
  AddEdgeInForward(graph, node, topo_indices);
  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

void AddControlEdgeForMemorySwapRewriter::AddEdgeInForward(Graph& graph, Node& node, const std::unordered_map<NodeIndex, int>& topo_indices) const {
  // SwapToHost is in forward, need to make sure it happens as early as possible
  // find the input node (src_node) to SwapToHost, and then find its output node taking the same input as SwapToHost
  // sometimes there might be no node taking the same input as SwapToHost, e.g. saved_mean in LayerNorm
  // in that case, we just find any output of the src_node to SwapToHost
  ORT_ENFORCE(node.GetInputEdgesCount() == 1);
  const auto& src_edge = *(node.InputEdgesBegin());
  const auto& src_node = src_edge.GetNode();
  const auto& src_arg_idx = src_edge.GetSrcArgIndex();

  NodeIndex node_idx = node.Index();
  int min_topo_index = INT_MAX;
  NodeIndex node_found = 0;
  int min_arg_topo_index = INT_MAX;
  NodeIndex arg_node_found = 0;
  for (auto iter = src_node.OutputEdgesBegin(); iter != src_node.OutputEdgesEnd(); ++iter) {
    const Node& peer_node = iter->GetNode();
    if (peer_node.OpType() == "SwapToHost")
      continue;

    int topo_index = topo_indices.at(peer_node.Index());
    if (iter->GetSrcArgIndex() == src_arg_idx) {
      if (topo_index < min_topo_index) {
        min_topo_index = topo_index;
        node_found = peer_node.Index();
      }
    } else if (!IsBackwardNode(iter->GetNode())) {
      if (topo_index < min_arg_topo_index) {
        min_arg_topo_index = topo_index;
        arg_node_found = peer_node.Index();
      }
    }
  }
  // add new edge to enforce swap node order, and update precedences
  if (min_topo_index < INT_MAX) {
    graph.AddControlEdge(node_idx, node_found);
  } else if (min_arg_topo_index < INT_MAX) {
    graph.AddControlEdge(node_idx, arg_node_found);
  } else {
    // there could be some optimizations making src_node no longer needed in FW, while used in BW
    // so remove swap nodes from src_node, and link src_node directly to dst_node
    const auto& swap_in = *node.OutputNodesBegin();
    Node::EdgeSet swap_in_output_edges(swap_in.OutputEdgesBegin(), swap_in.OutputEdgesEnd());
    for (auto edge : swap_in_output_edges) {
      graph.RemoveEdge(swap_in.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
      graph.AddEdge(src_node.Index(), edge.GetNode().Index(), src_arg_idx, edge.GetDstArgIndex());
    }
    graph.RemoveNode(swap_in.Index());
    graph.RemoveNode(node_idx);
  }
}

}  // namespace onnxruntime
