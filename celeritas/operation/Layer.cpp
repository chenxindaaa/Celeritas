#include "Layer.h"

namespace eCEL {

// void RMSNormLayer::forward(const ForwardContext& ctx,
//                            const Tensor& input,
//                            Tensor& output) {
//     (void)ctx;
//     (void)input;
//     (void)output;
// }

// void SelfAttentionLayer::forward(const ForwardContext& ctx,
//                                  const Tensor& input,
//                                  Tensor& output) {
//     (void)ctx;
//     (void)input;
//     (void)output;
// }

// void MLPBlockLayer::forward(const ForwardContext& ctx,
//                             const Tensor& input,
//                             Tensor& output) {
//     (void)ctx;
//     (void)input;
//     (void)output;
// }

// void DecoderLayer::forward(const ForwardContext& ctx,
//                            const Tensor& input,
//                            Tensor& output) {
//     Tensor attn_norm_output = input;
//     Tensor attn_output = input;
//     Tensor ffn_norm_output = input;

//     attn_norm_.forward(ctx, input, attn_norm_output);
//     self_attn_.forward(ctx, attn_norm_output, attn_output);
//     ffn_norm_.forward(ctx, attn_output, ffn_norm_output);
//     mlp_.forward(ctx, ffn_norm_output, output);
// }

}  // namespace eCEL
