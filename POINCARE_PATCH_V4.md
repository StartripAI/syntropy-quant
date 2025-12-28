# 庞加莱补丁 v4.0 (The Poincaré Patch)

## 🔧 核心修复

### 1. 消除 NaN 奇点
- **问题**: `High == Low` 导致除零错误，产生 NaN
- **修复**: 引入 `epsilon = 1e-8` 稳定化所有除法运算
- **位置**: `src/data/features.py`

### 2. 负阻尼解锁
- **问题**: 强制 `γ > 0` 导致模型对抗趋势（如 NVDA 的 AI 泡沫）
- **修复**: 自适应阻尼 `γ(state)` 允许负值，捕捉能量自注入
- **位置**: `src/core/physics.py`

### 3. 半隐式辛积分器
- **问题**: 显式欧拉法在非线性市场中数值发散
- **修复**: 升级为半隐式辛积分器，保持相空间体积守恒
- **位置**: `src/core/physics.py`

### 4. 数据源优化
- **问题**: Polygon API 403/429 错误导致训练失败
- **修复**: 优先使用 Yahoo Finance，Tiingo 作为备用
- **位置**: `src/data/fetcher.py`

## 📊 预期改进

1. **Loss < 1.0**: 模型开始学习（之前 Loss ≈ 1.25 表示最大熵状态）
2. **NVDA 正收益**: 负阻尼允许模型乘势而上
3. **无 RuntimeWarning**: 所有数值奇点已消除

## 🚀 使用方法

```bash
export TIINGO_API_KEY="your_key"

# 训练（等待 Yahoo Finance 限流解除）
python3 train_kernel.py \
  --epochs 40 \
  --lr 0.001 \
  --symbols "SPY,QQQ,IWM,NVDA,AAPL,MSFT,GOOGL,AMZN,META,TSLA,LLY,UNH,WMT"

# 回测
python3 run_backtest.py --thresh 0.1
```

## ⚠️ 注意事项

- Yahoo Finance 有速率限制，如果遇到 `YFRateLimitError`，请等待几分钟后重试
- 建议分批训练，每次 3-5 个符号
- 可以使用缓存数据（如果之前下载过）

## 📝 数学原理

### 负阻尼物理意义

$$ \frac{dp}{dt} = -\frac{\partial H}{\partial q} - \gamma p $$

- **γ > 0**: 能量耗散 → 均值回归
- **γ < 0**: 能量注入 → 趋势/泡沫

### 半隐式辛积分

使用 `p_new` 更新 `q`，确保辛结构保持：

```python
p_new = p + (force - gamma * p) * dt
q_new = q + velocity(p_new) * dt
```

---

**版本**: 4.0.0  
**日期**: 2025-12-27  
**作者**: StartripAI

