# src/algorithm/generate.jl

"""
    generate(model, ps, tokenizer, prompt; max_new_tokens=50, temperature=1.0)

自回归文本生成。
"""
function generate(model, ps, tokenizer, prompt::String; 
                  max_new_tokens::Int=50, temperature::Float32=1.0f0, block_size::Int=32)
    
    # 1. 编码 Prompt
    # ids: Vector{Int}
    input_ids = encode(tokenizer, prompt)
    
    # 确保不为空
    if isempty(input_ids)
        input_ids = [1] # fallback
    end

    for _ in 1:max_new_tokens
        # 2. 截断上下文 (不能超过 block_size)
        # 如果当前序列太长，只取最后 block_size 个 token
        cond_idx = max(1, length(input_ids) - block_size + 1)
        idx_cond = input_ids[cond_idx:end]
        
        # 3. 准备输入 (Seq, Batch=1)
        # reshape 为 (Seq, 1) 以适配 src/algorithm/train.jl 中的维度逻辑
        x = reshape(idx_cond, :, 1)
        
        # 4. 前向传播
        # model 返回 SpatialTensor{1}, .data 为 (Vocab, Seq, 1)
        # 我们这里手动包装一下或者让 Embed 处理，根据 src/module/embed.jl，Embed 可以直接处理 Array
        # 但为了通过 Sequential 的 _check 或保持一致性，我们传入 Matrix{Int}
        
        logits = model(x, ps).data
        
        # 5. 取最后一个时间步的预测 (Predict Next Token)
        # (Vocab, 1)
        next_token_logits = logits[:, end, 1] ./ temperature
        
        # 6. 采样 (Sampling)
        # 为了简单，这里用贪婪采样 (Greedy): 直接取最大值
        next_token = argmax(next_token_logits)
        
        # 7. 拼接
        push!(input_ids, next_token)
    end
    
    # 8. 解码回文本
    return decode(tokenizer, input_ids)
end