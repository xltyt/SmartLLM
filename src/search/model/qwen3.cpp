#include "qwen3.h"
#include "safetensors.hh"
#include <fstream>
#include <vector>
#include "utils/model_utils.h"

// ============ Attention ============
Qwen3Attention::Qwen3Attention(int hidden_size, int num_heads, int num_kv_heads, int head_dim, float rms_norm_eps) {
  //int head_dim = hidden_size / num_heads;
  //LOG(INFO) << "Qwen3Attention hidden_size[" << hidden_size << "] num_heads[" << num_heads << "] head_dim[" << head_dim << "]";
  this->q_proj = register_module("q_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, num_heads * head_dim).bias(false)
    ));
  this->k_proj = register_module("k_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, num_kv_heads * head_dim).bias(false)
    ));
  this->v_proj = register_module("v_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, num_kv_heads * head_dim).bias(false)
    ));
  this->o_proj = register_module("o_proj", torch::nn::Linear(
    torch::nn::LinearOptions(num_heads * head_dim, hidden_size).bias(false)
    ));
  this->q_norm = register_module("q_norm", std::make_shared<RMSNorm>(head_dim, rms_norm_eps));
  this->k_norm = register_module("k_norm", std::make_shared<RMSNorm>(head_dim, rms_norm_eps));
  _head_dim = head_dim;
  _scaling = std::pow(head_dim, -0.5);
  _num_key_value_groups = num_heads / num_kv_heads;
}

Qwen3Attention::~Qwen3Attention() {
}

// Rotates half the hidden dims of the input.
inline torch::Tensor rotate_half(const torch::Tensor& x) {
    int64_t half_dim = x.size(-1) / 2;
    auto x1 = x.slice(-1, 0, half_dim);
    auto x2 = x.slice(-1, half_dim);
    return torch::cat({-x2, x1}, /*dim=*/-1);
}

//
//  Applies Rotary Position Embedding to the query and key tensors.
//
//  Args:
//      q (`torch.Tensor`): The query tensor.
//      k (`torch.Tensor`): The key tensor.
//      cos (`torch.Tensor`): The cosine part of the rotary embedding.
//      sin (`torch.Tensor`): The sine part of the rotary embedding.
//      unsqueeze_dim (`int`, *optional*, defaults to 1):
//          The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
//          sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
//          that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
//          k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
//          cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
//          the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
//  Returns:
//      `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
//

inline std::pair<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& cos,
    const torch::Tensor& sin) 
{
    // cos, sin 形状通常为 [batch, 1, seq_len, head_dim] 或 [1, 1, seq_len, head_dim]
    auto q_embed = (query * cos) + (rotate_half(query) * sin);
    auto k_embed = (key * cos) + (rotate_half(key) * sin);
    return {q_embed, k_embed};
}

// This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
// num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
inline torch::Tensor repeat_kv(const torch::Tensor& hidden_states, int64_t n_rep) {
    if (n_rep == 1) {
        return hidden_states;
    }
    // 输入维度: [batch, num_key_value_heads, slen, head_dim]
    int64_t batch = hidden_states.size(0);
    int64_t num_kv_heads = hidden_states.size(1);
    int64_t slen = hidden_states.size(2);
    int64_t head_dim = hidden_states.size(3);

    // unsqueeze(2) 扩展并在第 2 维 expand，最后 reshape 合并维度
    return hidden_states.unsqueeze(2)
        .expand({batch, num_kv_heads, n_rep, slen, head_dim})
        .reshape({batch, num_kv_heads * n_rep, slen, head_dim});
}

inline std::pair<torch::Tensor, torch::Tensor> eager_attention_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int64_t num_key_value_groups,
    const std::optional<torch::Tensor>& attention_mask = std::nullopt,
    double scaling = 0.0,
    double dropout = 0.0,
    bool is_training = false)
{
    // 如果没有传入 scaling，默认使用 1.0 / sqrt(head_dim)
    if (scaling <= 0.0) {
        scaling = 1.0 / std::sqrt(static_cast<double>(query.size(-1)));
    }

    // 1. 对 key 和 value 进行 GQA 头维度扩展
    // 变换后维度: [batch, num_heads, kv_len, head_dim]
    auto key_states = repeat_kv(key, num_key_value_groups);
    auto value_states = repeat_kv(value, num_key_value_groups);

    // 2. 计算 Attention 原始权重: Q · K^T * scaling
    // query: [B, N, Q_len, D]
    // key_states.transpose(2, 3): [B, N, D, KV_len]
    // attn_weights: [B, N, Q_len, KV_len]
    auto attn_weights = torch::matmul(query, key_states.transpose(2, 3)) * scaling;

    // 3. 叠加 Attention Mask (通常在 Prefill 时包含负无穷 -inf)
    if (attention_mask.has_value() && attention_mask->defined()) {
        attn_weights = attn_weights + attention_mask.value();
    }

    // 4. 数值稳定 Softmax：强制以 FP32 计算指数与累加，然后转回原始数据类型
    attn_weights = torch::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(query.dtype());

    // 5. Dropout (推理时 dropout=0.0 或 is_training=false，底层会自动跳过)
    if (dropout > 0.0 && is_training) {
        attn_weights = torch::dropout(attn_weights, dropout, is_training);
    }

    // 6. 加权求和得到输出: attn_weights · V
    // value_states: [B, N, KV_len, D]
    // attn_output: [B, N, Q_len, D]
    auto attn_output = torch::matmul(attn_weights, value_states);

    // 7. 转置并确保内存连续: [B, N, Q_len, D] -> [B, Q_len, N, D]
    attn_output = attn_output.transpose(1, 2).contiguous();

    return {attn_output, attn_weights};
}

std::tuple<torch::Tensor, torch::Tensor> Qwen3Attention::forward(
  const torch::Tensor& hidden_states,
  const std::tuple<torch::Tensor, torch::Tensor>& position_embeddings,
  const std::optional<torch::Tensor>& attention_mask
  ) {

  int64_t batch_size = hidden_states.size(0);
  int64_t seq_len = hidden_states.size(1);
  std::vector<int64_t> hidden_shape = {batch_size, seq_len, -1, _head_dim};

  auto q_projected = this->q_proj->forward(hidden_states).view(hidden_shape);
  auto query_states = this->q_norm->forward(q_projected).transpose(1, 2);

  auto k_projected = this->k_proj->forward(hidden_states).view(hidden_shape);
  auto key_states = this->k_norm->forward(k_projected).transpose(1, 2);
  //auto key_states = this->k_proj->forward(hidden_states).view(hidden_shape).transpose(1, 2);

  auto value_states = this->v_proj->forward(hidden_states).view(hidden_shape).transpose(1, 2);
  
  const torch::Tensor& position_embeddings_cos = std::get<0>(position_embeddings);
  const torch::Tensor& position_embeddings_sin = std::get<1>(position_embeddings);
  //const torch::Tensor& position_embeddings_sin = std::get<0>(position_embeddings);

  std::tie(query_states, key_states) = apply_rotary_pos_emb(
    query_states,
    key_states,
    position_embeddings_cos,
    position_embeddings_sin
    );
        
  auto [attn_output, attn_weights] = eager_attention_forward(
    query_states,
    key_states,
    value_states,
		_num_key_value_groups,
    attention_mask,
    _scaling,
    0.0,
		false
  );

  attn_output = attn_output.reshape({batch_size, seq_len, -1}).contiguous();
  attn_output = this->o_proj(attn_output);
  return {attn_output, attn_weights};
}

// ============ MLP ============
//
// See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
// Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
// Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
// Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
// later.
//
class SiLUActivation : public torch::nn::Module {
public:
  torch::Tensor forward(const torch::Tensor& input) {
    return torch::nn::functional::silu(input);
  }
};

Qwen3MLP::Qwen3MLP(int hidden_size, int intermediate_size) {
  this->gate_proj = register_module("gate_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)
    ));
  this->up_proj = register_module("up_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)
    ));
  this->down_proj = register_module("down_proj", torch::nn::Linear(
    torch::nn::LinearOptions(intermediate_size, hidden_size).bias(false)
    ));
}

Qwen3MLP::~Qwen3MLP() {
}
  
torch::Tensor Qwen3MLP::forward(const torch::Tensor& x) {
  SiLUActivation silu;
  return this->down_proj(silu.forward(this->gate_proj(x)) * this->up_proj(x));
}

// ============ Decoder Layer ============
Qwen3DecoderLayer::Qwen3DecoderLayer(int hidden_size, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps) {
  this->input_layernorm = register_module("input_layernorm", std::make_shared<RMSNorm>(hidden_size, rms_norm_eps));
  this->self_attn = register_module("self_attn", std::make_shared<Qwen3Attention>(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps));
  this->post_attention_layernorm = register_module("post_attention_layernorm", std::make_shared<RMSNorm>(hidden_size, rms_norm_eps));
  this->mlp = register_module("mlp", std::make_shared<Qwen3MLP>(hidden_size, intermediate_size));
}

Qwen3DecoderLayer::~Qwen3DecoderLayer() {
}

torch::Tensor Qwen3DecoderLayer::forward(
  const torch::Tensor& hidden_states,
  const std::tuple<torch::Tensor, torch::Tensor>& position_embeddings,
  const std::optional<torch::Tensor>& attention_mask
  ) {
  //LOG(INFO) << "Qwen3DecoderLayer::forward layer_norm Input [" << format_tensor(hidden_states) << "]";
  auto hidden_states_new = this->input_layernorm->forward(hidden_states);
  //LOG(INFO) << "Qwen3DecoderLayer::forward layer_norm Input2 [" << format_tensor(hidden_states) << "]";
  //LOG(INFO) << "Qwen3DecoderLayer::forward layer_norm Output [" << format_tensor(hidden_states_new) << "]";
  
  // Self Attention
  //LOG(INFO) << "Qwen3DecoderLayer::forward self_attn hidden_states_new [" << format_tensor(hidden_states_new) << "]";
  //LOG(INFO) << "Qwen3DecoderLayer::forward self_attn position_embeddings_cos [" << format_tensor(std::get<0>(position_embeddings)) << "]";
  //LOG(INFO) << "Qwen3DecoderLayer::forward self_attn position_embeddings_sin [" << format_tensor(std::get<1>(position_embeddings)) << "]";
  //LOG(INFO) << "Qwen3DecoderLayer::forward self_attn attention_mask [" << format_tensor(attention_mask.value()) << "]";
  torch::Tensor attn_weights;
  std::tie(hidden_states_new, attn_weights) = this->self_attn->forward(
    hidden_states_new,
    position_embeddings,
    attention_mask
    );
  //LOG(INFO) << "Qwen3DecoderLayer::forward self_attn attn_output [" << format_tensor(hidden_states_new) << "]";
  //LOG(INFO) << "Qwen3DecoderLayer::forward self_attn attn_weights [" << format_tensor(attn_weights) << "]";
  //LOG(INFO) << "Qwen3DecoderLayer::forward post_norm ori [" << format_tensor(hidden_states) << "]";
  hidden_states_new = hidden_states + hidden_states_new;
  
  // Fully Connected
  auto residual = hidden_states_new;
  //LOG(INFO) << "Qwen3DecoderLayer::forward post_norm Input [" << format_tensor(hidden_states_new) << "]";
  hidden_states_new = this->post_attention_layernorm->forward(hidden_states_new);
  //LOG(INFO) << "Qwen3DecoderLayer::forward post_norm Output [" << format_tensor(hidden_states_new) << "]";
  hidden_states_new = this->mlp->forward(hidden_states_new);
  hidden_states_new = residual + hidden_states_new;
  return hidden_states_new;
}

// ============ RotaryEmbedding ============
Qwen3RotaryEmbedding::Qwen3RotaryEmbedding(
  int64_t head_dim,
  double rope_theta /*= 1000000.0*/,
  torch::Device device /*= torch::kCPU*/) {
        
  auto [computed_inv_freq, factor] = compute_default_rope_parameters(head_dim, rope_theta, device);
  _attention_scaling = factor;

  _inv_freq = register_buffer("inv_freq", computed_inv_freq);
  _original_inv_freq = register_buffer("original_inv_freq", computed_inv_freq.clone());
}

std::tuple<torch::Tensor, double> Qwen3RotaryEmbedding::compute_default_rope_parameters(
  int64_t head_dim,
  double rope_theta,
  torch::Device device /*= torch::kCPU*/) {
  
  auto arange_tensor = torch::arange(0, head_dim, 2, torch::device(device).dtype(torch::kInt64));

  auto exponent = arange_tensor.to(torch::kFloat32) / static_cast<double>(head_dim);

  auto inv_freq = 1.0 / torch::pow(rope_theta, exponent);
  
  double attention_factor = 1.0;
  return std::make_tuple(inv_freq, attention_factor);
}

std::pair<torch::Tensor, torch::Tensor> Qwen3RotaryEmbedding::forward(
  const torch::Tensor& x,
  const torch::Tensor& position_ids
  ) {
  torch::NoGradGuard no_grad;

  int64_t batch_size = position_ids.size(0);
  auto inv_freq_expanded = _inv_freq
    .unsqueeze(0).unsqueeze(-1)
    .to(torch::kFloat32)
    .expand({batch_size, -1, 1})
    .to(x.device());

  auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);

  auto freqs = torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);

  auto emb = torch::cat({freqs, freqs}, /*dim=*/-1);

  auto cos = (emb.cos() * _attention_scaling).to(x.dtype());
  auto sin = (emb.sin() * _attention_scaling).to(x.dtype());

  return {cos, sin};
}

// ============ Model ============

inline torch::Tensor make_causal_mask(
  int64_t seq_len,
  torch::Dtype dtype = torch::kFloat32,
  torch::Device device = torch::kCPU
  ) {
  // 1. 在 Softmax 中使用 -1e4f 或 -1e9f 足以让概率彻底归零，同时避免极端下溢
  constexpr float MIN_VAL = -1e4f;

  // 2. 创建 seq_len x seq_len 的常数矩阵
  auto mask = torch::full({seq_len, seq_len}, MIN_VAL, torch::dtype(dtype).device(device));

  // 3. triu(1) 仅保留对角线上方为 MIN_VAL，其余位置清零
  mask = torch::triu(mask, /*diagonal=*/1);

  // 4. unsqueeze 扩展至 [1, 1, seq_len, seq_len]
  return mask.unsqueeze(0).unsqueeze(0);
}

Qwen3Model::Qwen3Model(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps) {
  this->embed_tokens = register_module("embed_tokens", torch::nn::Embedding(vocab_size, hidden_size));

  this->layers = register_module("layers", torch::nn::ModuleList());
  for (int i = 0; i < num_layers; i++) {
    this->layers->push_back(std::make_shared<Qwen3DecoderLayer>(
      hidden_size,
      num_heads,
      num_kv_heads,
      head_dim,
      intermediate_size,
      rms_norm_eps
      ));
  }

  this->norm = register_module("norm", std::make_shared<RMSNorm>(hidden_size, rms_norm_eps));
    
  this->rotary_emb = std::make_shared<Qwen3RotaryEmbedding>(head_dim);
}

Qwen3Model::~Qwen3Model() {
}

std::tuple<torch::Tensor, torch::Tensor> Qwen3Model::prepare(
  const std::vector<int64_t>& input_ids
  ) {

  torch::Tensor tensor_input_ids = torch::from_blob(
    (void *)input_ids.data(), 
    {1, static_cast<int64_t>(input_ids.size())}, 
    torch::kInt64
    );
  auto inputs_embeds = this->embed_tokens->forward(tensor_input_ids);

  int64_t seq_len = inputs_embeds.size(1);

  auto position_ids = torch::arange(seq_len, torch::device(inputs_embeds.device()).dtype(torch::kInt64)) + 0;

  position_ids = position_ids.unsqueeze(0);

  return std::make_tuple(inputs_embeds, position_ids);
}

torch::Tensor Qwen3Model::forward(
  const std::vector<int64_t>& input_ids
  ) {
  
  auto [hidden_states, position_ids] = prepare(input_ids);
  
  auto position_embeddings = this->rotary_emb->forward(hidden_states, position_ids);
  
  auto attention_mask = make_causal_mask(input_ids.size());
  
  for (int i = 0; i < this->layers->size(); i++) {
    Qwen3DecoderLayer *layer = (Qwen3DecoderLayer *)this->layers[i].get();
    hidden_states = layer->forward(hidden_states, position_embeddings, attention_mask);
  }
  hidden_states = this->norm->forward(hidden_states);
  
  return hidden_states;
}

// ============ Top-level Model ============
Qwen3ForCausalLM::Qwen3ForCausalLM::Qwen3ForCausalLM(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps) {
  this->model = register_module("model", std::make_shared<Qwen3Model>(
    vocab_size,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    rms_norm_eps
    ));

  this->lm_head = register_module("lm_head", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)
    ));
}

Qwen3ForCausalLM::~Qwen3ForCausalLM() {
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
