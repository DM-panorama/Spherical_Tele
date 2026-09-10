# TeleStyle 南北 Chart 联合 Decoder 方案

## 1. 背景与目标

当前半球模式分别对 North/South stereographic chart 去噪，再使用原始 Qwen VAE decoder 分别解码，最后重投影为 ERP，并在赤道处选择不同分支的 RGB。即使两条去噪轨迹在重叠区域已经进行过特征交互，最终仍可能因以下差异产生赤道接缝：

- 两个分支的亮度、色调或对比度不同；
- 笔触、高频纹理或噪声相位不同；
- 物体边缘存在轻微位置偏差；
- 两次独立 VAE 解码放大了 latent 中的微小差异；
- ERP 合成在赤道处硬切换 RGB 来源，没有后续网络修复边界。

本方案引入 **North/South Chart Joint Decoder**。它保留现有 Qwen/TeleStyle latent 空间和大部分预训练 VAE decoder，只在 decoder 中加入球面重投影、投影可靠度融合和少量共享 ERP 解码层，最终通过一个 RGB head 一次生成完整 ERP。

方案借鉴 SphereDiff 的核心原则，但不迁移其 Fibonacci 球面点云表示：

> 对同一球面位置的多个投影视图结果，不按固定边界选择，而应根据各投影在该位置的几何可靠度进行归一化融合。

SphereDiff 将该原则用于多个透视 latent patch 的逐步去噪和最终合成；本方案将它用于 North/South decoder feature，并在融合后继续通过 shared ERP decoder 重新生成 RGB。这样既保留 Qwen VAE 的规则二维 latent，又避免停留在多 patch RGB blending。

核心目标是：

1. 取消 North RGB 与 South RGB 的赤道硬切；
2. 根据投影可靠度，在多通道 feature 空间协调两条分支的差异；
3. 让赤道上下文经过共享卷积后再生成 RGB；
4. 尽量保持原始 Qwen VAE 的 latent 解释和非接缝区域画质；
5. 不重新训练 encoder、DiT 或完整 VAE。

联合 decoder 消除的是“RGB 硬切”这一结构性接缝来源。若两条 latent 的物体结构严重不一致，仍可能出现重影或错位，因此去噪阶段的 SphereAdapter 与最终联合 decoder 是互补关系。

## 2. 总体思路

### 2.1 当前流程

```text
z_N → Original VAE Decoder → North RGB ┐
                                        ├→ ERP reprojection → equator hard cut
z_S → Original VAE Decoder → South RGB ┘
```

赤道是两张已经完成生成的 RGB 图像之间的边界，硬切后没有网络可以继续修复它。

### 2.2 联合解码流程

```text
z_N [B,16,h,h]                    z_S [B,16,h,h]
       │                                  │
       ├──── shared-weight chart stem ────┤
       │                                  │
     F_N [B,C,r,r]                      F_S [B,C,r,r]
       │                                  │
       └──── stereographic → ERP sampling ┘
                         │
              F_N^erp, F_S^erp [B,C,r/2,r]
                         │
       validity mask + projection-reliability confidence
                         │
              learnable residual feature fusion
                         │
              shared ERP feature [B,C,r/2,r]
                         │
          shared ERP residual/upsample decoder blocks
                         │
                    one RGB head
                         │
                  ERP [B,3,H,2H]
```

North/South chart stem 使用同一套预训练权重。两路 feature 被球面重投影到相同 ERP 网格后连续融合；从第一次融合开始，后续 decoder 只保留一条共享 ERP 分支。

### 2.3 与 SphereDiff 的关系

SphereDiff 的完整做法是维护一组近似均匀分布的 Fibonacci spherical latent，在每个 timestep 将不同观察方向覆盖的球面点排列成规则透视 latent patch，交给预训练平面 diffusion model 去噪，再按投影可靠度回写到统一球面状态。最终阶段仍然解码多个透视 patch，并在 RGB 空间加权合成 ERP。

本方案只迁移其中最适合 TeleStyle 的一层抽象：

```text
SphereDiff:
same spherical position
  → predictions from multiple perspective projections
  → projection-reliability weighted fusion

TeleStyle Joint Decoder:
same spherical position
  → North/South decoder features
  → projection-reliability weighted fusion
  → shared ERP decoder
  → one RGB output
```

明确不迁移的部分包括：

- 不使用 Fibonacci point latent；
- 不改变 Qwen DiT 的规则二维 latent 定义；
- 不将球面点动态重排为大量透视 patch；
- 不在最终阶段分别解码多个 patch 后进行 RGB blending。

因此，这是一种受 SphereDiff 启发、但针对现有双 stereographic chart 架构重新设计的联合 decoder，而不是 SphereDiff 的直接移植。

## 3. 推荐的第一版结构

### 3.1 采用中后段单点融合

第一版不直接实现复杂的全多尺度 attention，而是在 decoder 中后段选择一个融合点：

1. `post_quant_conv`、`conv_in`、bottleneck 和前若干 upsample block 分别处理两张 chart；
2. 两个 chart stem 严格共享参数；
3. 在选定分辨率通过确定性的球面 correspondence 将两路特征重投影到 ERP；
4. 使用 SphereDiff 式投影可靠度和轻量可学习残差模块融合；
5. 增加 2～3 个 shared ERP residual block；
6. 复用 decoder 剩余 upsample block、`norm_out` 和 `conv_out`，一次输出 ERP RGB。

建议第一版从倒数第二个 upsample block 前或后做实验。融合太晚时，共享卷积感受野较小；融合太早时，feature 重采样会影响更大范围的内容。最终位置由消融实验决定。

### 3.2 形状约定

若最终 chart RGB 尺寸为 `[S,S]`，对应目标 ERP 尺寸为 `[S/2,S]`。在任一 decoder stage：

```text
chart feature: [B,C,r,r]
ERP feature:   [B,C,r/2,r]
```

后续普通上采样同时扩大高度和宽度，最终自然恢复为 2:1 ERP。所有目标尺寸必须满足当前 DiT/VAE 的 8 倍缩放和仓库要求的 16 像素对齐规则。

### 3.3 球面 correspondence

对于 ERP feature 像素中心 `p`：

1. 将 ERP 像素中心转换为球面单位方向 `d(p)`；
2. 分别投影到 North/South stereographic chart，得到 `u_N(p)` 和 `u_S(p)`；
3. 使用 `grid_sample` 从两张 feature map 采样；
4. 同时生成每张 chart 的有效域 mask、投影半径和局部投影可靠度。

对应关系应复用仓库现有的球面坐标约定，保持 South chart 的 yaw、像素中心和 `align_corners` 语义在训练与推理中完全一致。禁止用普通二维旋转或简单翻转代替球面重投影。

这里的核心不是判断一个点位于北纬还是南纬，而是判断同一球面位置从哪个 chart 观察更可靠。纬度可以参与有效域和所有权定义，但不能作为唯一融合依据。

### 3.4 SphereDiff 式投影可靠度融合

基础融合为：

```math
f(p)=w_N(p)f_N(p)+w_S(p)f_S(p), \qquad w_N(p)+w_S(p)=1
```

对 chart `i` 和 ERP 位置 `p`，先计算该点在 stereographic chart 上距投影中心的归一化半径：

```math
\rho_i(p)=\left\|u_i(p)\right\|
```

第一版借鉴 SphereDiff 的中心优先思想，采用指数衰减的几何置信度：

```math
q_i(p)=m_i(p)\exp\left(-\frac{\rho_i(p)}{\tau}\right)
```

```math
w_i^{geo}(p)=\frac{q_i(p)}{q_N(p)+q_S(p)+\epsilon}
```

其中越靠近 chart 投影中心，可靠度越高；越靠近投影边缘，可靠度越低。这里迁移的是 SphereDiff 的原则，具体函数必须针对 stereographic 投影重新校准，不能直接照搬其 perspective patch 的 `tau`。后续可以用 stereographic 投影 Jacobian、局部角分辨率或重采样放大率替代单一半径，形成更准确的可靠度：

```math
q_i(p)=m_i(p)\,g\left(\rho_i(p),J_i(p),a_i(p)\right)
```

权重最终由三部分组成：

- `m_N, m_S`：chart 有效域 mask；
- `w_geo`：由投影半径或 Jacobian 得到的确定性几何可靠度；
- `delta_l`：根据 `f_N`、`f_S`、球面位置和二者差异预测的可学习 logit 修正。

推荐形式：

```math
\ell_i=\log(q_i+\epsilon)+\alpha\Delta\ell_i,\qquad
(w_N,w_S)=\operatorname{masked\_softmax}(\ell_N,\ell_S)
```

其中 `delta_l` 输出层和标量门 `alpha` 均零初始化。训练开始时模型严格退化为可解释的投影可靠度融合，随后才逐渐学习内容相关的修正。使用 masked softmax 可以保证无效 chart 权重为零、有效权重非负且总和为一。学习分支只允许修正几何先验，不应完全覆盖它。

仅使用加权平均可能在结构错位处产生双影，因此融合模块还应预测一个零初始化的残差：

```math
F_{shared}=w_NF_N^{erp}+w_SF_S^{erp}
             +\beta R(F_N^{erp},F_S^{erp},F_N^{erp}-F_S^{erp},p)
```

`beta` 同样从零开始。`R` 第一版可使用 2～3 个小型 residual block，不必立即采用高成本全局 attention。

### 3.5 ERP 拓扑处理

共享 ERP decoder 至少在水平方向使用循环 padding，使左右边界互为邻居。垂直方向的越极点 padding 可后续加入：

- 纬度越过极点后反射；
- 经度平移半周；
- 保留 Qwen `CausalConv3d` 时间维的原始 causal padding，只改变空间 H/W padding。

第一阶段优先解决赤道融合和水平环绕，极点专用 padding 不作为验证联合 decoder 是否有效的前置条件。

## 4. 参数初始化与冻结策略

### 4.1 必须保持不变的部分

- 保持原始 16 通道 latent、均值、标准差和缩放规则；
- 冻结 VAE encoder；
- 不改变 DiT 输出定义；
- North/South 使用同一个 chart decoder stem；
- 不为两张 chart 分别训练两套 decoder 权重。

### 4.2 第一阶段可训练参数

- feature fusion 的权重预测器；
- 零初始化 residual fusion block；
- 新增 shared ERP residual block；
- 可选的球面位置投影分支；
- 少量标量 gate。

原 decoder stem 初始冻结。原 decoder 后半段与 RGB head 可以先冻结，再用远小于新增模块的学习率解冻最后一个 block。

如果加入 `(x,y,z)` 球面位置，不直接扩充预训练卷积的输入通道，而采用零初始化旁路：

```text
feature = pretrained_block(feature) + zero_init_position_proj(xyz)
```

这保证初始化时输出与原 decoder 尽可能接近。

## 5. 训练方法

### 5.1 阶段 A：几何与重建预训练

目的：让联合 decoder 学会正确重投影、融合并重建同一真实场景。

```text
真实 ERP
  → 随机球面旋转
  → 提取重叠 North/South chart
  → 冻结的原 VAE encoder
  → clean z_N, clean z_S
  → Joint Decoder
  → predicted ERP
  → 与旋转后的真实 ERP 监督
```

训练策略：

- 冻结 encoder、DiT 和 chart stem；
- 训练 fusion 与 shared ERP blocks；
- 首先固定解析的投影可靠度，仅训练零初始化 residual；稳定后再开放小幅 reliability logit 修正；
- 在验证集上校准 `tau`，并与 stereographic Jacobian 权重比较；
- 以原 decoder 重建为非接缝区域教师；
- 从固定 overlap 和固定尺寸开始，稳定后再增加变化；
- 不把普通扩散噪声 `z_t` 直接送入 decoder。

阶段 A 验证模型能否保持基本 VAE 重建质量，但干净 latent 对天然高度一致，不能单独证明它能修复实际生成接缝。

### 5.2 阶段 B：TeleStyle 生成 latent 分布适配

目的：学习推理时真实存在的南北差异，是解决实际接缝的关键阶段。

离线运行当前 TeleStyle/SphereAdapter 流程并缓存：

- 最终 `z_N`、`z_S`；
- 内容 ERP 或内容 chart；
- style 标识与 prompt；
- seed、overlap、chart size；
- 当前硬切结果、两张独立 decoder RGB；
- 可选的去噪过程 `x0` 估计。

优先使用最终去噪 latent 或从噪声预测换算出的 `x0_hat`。任意 timestep 的 `z_t` 不属于 VAE decoder 输入分布；除非额外加入 timestep 条件并把 decoder 改成去噪器，否则不得直接用于训练。

若生成样本没有风格化真值 ERP，可组合使用：

- 非赤道区域的原 decoder 蒸馏；
- North/South overlap 的 feature/RGB 一致性；
- 输入内容的低频结构保持；
- 跨赤道透视视图的感知连续性；
- 人工接缝数据的已知无缝目标。

可对 clean latent 施加小幅扰动，但扰动统计应来自实际生成 latent 对的误差分布，而不是随意添加大幅高斯噪声。

### 5.3 阶段 C：小范围联合微调

在独立训练的 joint decoder 已经稳定后，可进一步：

1. 冻结 Qwen DiT 主干；
2. 同时训练 SphereAdapter 和 joint decoder；
3. SphereAdapter 负责低频结构与跨 chart 语义一致；
4. joint decoder 负责颜色、纹理和最终边界协调；
5. 如有必要，仅对原 decoder 最后一个 block 或 RGB head 使用低学习率。

不应从第一步就联合训练 DiT、SphereAdapter 和 decoder，否则难以定位质量退化来源，也容易改变原有 latent 语义。

## 6. 损失函数

总损失建议从以下四项起步：

```math
L=L_{recon}
 +\lambda_{eq}L_{equator}
 +\lambda_{per}L_{perspective}
 +\lambda_{distill}L_{distill}
 +\lambda_{reg}L_{gate}
```

### 6.1 球面加权重建损失

按 ERP 像素中心纬度使用面积权重：

```math
w(\phi)=\cos(\phi)
```

对 Charbonnier/L1 进行归一化加权，避免极区重复采样主导训练。阶段 A 使用真实 ERP；阶段 B 仅在存在可靠目标的区域使用。

### 6.2 赤道梯度损失

在赤道重叠带比较预测与目标的一阶梯度和可选 Laplacian：

```math
L_{equator}=\left\|\nabla\hat I-\nabla I\right\|_{|\phi|<d}
```

不要直接强迫赤道上下像素相同，因为真实场景可能有合法边缘。监督目标应是“预测跨赤道变化与目标一致”，而不是“赤道必须平坦”。

### 6.3 切平面感知损失

从预测和目标 ERP 渲染一组跨越赤道及左右接缝的透视视图，计算：

- Charbonnier；
- LPIPS；
- gradient loss。

该损失直接衡量正常观看视角中的连续性，降低 ERP 拉伸对指标的影响。

### 6.4 非接缝区域蒸馏

使用原 VAE decoder 分别解码两张 chart，并投影到 ERP，作为各自高置信区域的教师。远离赤道时强约束 joint decoder 保持原输出；接近赤道时降低蒸馏权重，让新 decoder 有空间进行修复。

### 6.5 环绕损失

ERP 首列和末列代表相邻像素中心而不是同一个采样点，因此不应简单要求二者数值完全相等。应比较：

- 预测与目标的跨边界有限差分；
- 跨左右接缝透视视图；
- 使用 circular padding 后的局部梯度。

## 7. 数据集要求

### 7.1 基础 ERP 数据

数据应满足：

- 标准 2:1 ERP；
- 最低高度建议 512，正式训练建议包含 512 和 1024 档；
- 无明显原始拼接缝、曝光断层或错误投影；
- 覆盖室内、室外、自然、城市和复杂高频纹理；
- 赤道附近包含直线、物体轮廓、人物、文字和规律纹理等困难内容；
- 训练、验证、测试按原始场景划分，旋转变体不得跨集合泄漏。

阶段 A 至少需要数千张高质量 ERP。小规模可行性实验可以先使用 500～1000 张，但不能据此判断泛化效果。

### 7.2 风格数据

应覆盖 TeleStyle 实际支持的风格范围，包括：

- 不同色调和对比度；
- 粗细不同的笔触；
- 高频线稿和低频绘画风格；
- 对结构保持要求不同的风格。

训练与验证必须包含未见 style，以检查 decoder 是否只记忆训练风格。style 图本身不直接作为 decoder 的额外输入；其影响已包含在生成 latent 中。

### 7.3 真实生成 latent 对

每个样本建议缓存：

```text
scene_id
source_erp / source_chart references
style_id
prompt_id
seed
overlap_degrees
chart_size
z_north
z_south
optional x0_hat_north / x0_hat_south
original north/south decoded RGB
current hard-cut ERP
model and checkpoint fingerprints
```

数据需要覆盖接缝严重程度的完整分布，不能只保存成功案例。建议根据赤道颜色差、梯度差和结构差分层采样，使训练集中包含：

- 无接缝或轻微接缝样本；
- 明显颜色/曝光接缝；
- 高频笔触不连续；
- 小幅几何错位；
- joint decoder 无法合理修复的严重结构冲突。

严重结构冲突样本用于识别能力边界，不应强迫 decoder 用模糊化掩盖所有错误。

### 7.4 数据增强

推荐增强：

- 随机 yaw，改变 ERP 左右接缝位置；
- 真实球面 pitch/roll 旋转；
- overlap 在约 12°～18° 内变化；
- North/South 交换及相应 geometry 交换；
- 从实测误差分布采样的小幅亮度、色调和 feature 扰动；
- 多输出分辨率训练或分阶段升分辨率。

所有 pitch/roll 增强必须通过球面单位方向和重采样完成，禁止使用普通二维旋转。

## 8. 训练与显存控制

联合 decoder 在融合前同时保留两条 activation，显存高于单次 VAE decode。建议：

- 冻结 chart stem，并在可能时缓存 clean latent 或中间 feature；
- 对 trainable shared blocks 使用 gradient checkpointing；
- 先以 512 chart 验证，再升到 1024；
- 使用 BF16；
- 保持 VAE 非 tiled decode 作为正确性基线，确认一致后再研究 tiled 训练；
- 使用梯度累积，不通过缩小随机 crop 破坏完整球面拓扑；
- 蒸馏教师分支放在 `no_grad` 下运行。

## 9. 评估与验收指标

### 9.1 必须保留的对照组

1. 当前 North/South RGB hard cut；
2. RGB cosine feathering；
3. final latent 重投影融合后单次原 VAE decode；
4. joint decoder，仅固定纬度权重；
5. joint decoder，固定投影半径权重；
6. joint decoder，投影权重 + learnable residual；
7. 可选的 stereographic Jacobian 权重；
8. 可选的多尺度 joint decoder。

这样可以区分收益来自“取消硬切”“单次解码”“投影可靠度”还是“学习型 feature fusion”。

### 9.2 定量指标

- 赤道带 RGB/Charbonnier 误差；
- 预测与目标的跨赤道一阶、二阶梯度误差；
- 跨赤道透视视图 LPIPS；
- 左右环绕透视视图 LPIPS；
- 非接缝区域相对原 decoder 的 LPIPS/PSNR；
- 赤道附近高频能量保持率；
- North/South 结构差较大时的重影指标或人工分级；
- 推理时间与峰值显存。

### 9.3 人工评测

采用隐藏 A/B 测试，重点检查：

- 赤道线是否仍可定位；
- 是否由硬接缝变成宽范围模糊带；
- 直线、文字和物体边缘是否出现双影；
- 风格笔触是否跨赤道连续；
- 非赤道区域是否发生不必要的颜色和细节变化；
- ERP 左右边界及极区是否退化。

## 10. 预期结果

### 10.1 合理预期

训练成功后，预期获得：

- 消除当前赤道单行 RGB 硬切；
- 显著降低轻度至中度亮度、色调和纹理接缝；
- 相比固定纬度羽化，更稳定地选择投影条件较好的 chart feature；
- 赤道附近笔触和局部梯度更连续；
- 非接缝区域基本保持原 VAE 画质；
- 与 SphereAdapter 配合时优于单独使用任一模块；
- 推理从两次完整 RGB decode 变为“两路部分 decode + 一路共享 decode”。

### 10.2 不应承诺的结果

联合 decoder 不能保证：

- 修复两条分支生成的完全不同物体或大幅几何错位；
- 同时彻底解决 ERP 极区拉伸；
- 在没有实际生成 latent 训练数据时泛化到所有 TeleStyle 风格；
- 在完全不增加计算量和显存的情况下获得上述收益。

最常见的失败形式可能从“清晰硬线”变为“赤道附近双影或模糊带”。因此验收不能只看硬 seam 数值下降，还必须检查高频能量和结构清晰度。

## 11. 实施里程碑

### M0：几何与接口验证

- 实现各 decoder stage 的 chart feature → ERP feature 映射；
- 验证像素中心、South yaw、有效域和输出形状；
- 对常量、经纬线和棋盘格做 CPU 单元测试；
- 确认梯度能通过 `grid_sample` 回传。

### M1：无训练基线

- 在 RGB head 前重投影 feature；
- 分别实现固定 latitude confidence、投影半径 reliability 和 Jacobian reliability；
- 对同一球面位置归一化 North/South 几何权重；
- 复用原 RGB head 输出 ERP；
- 与 RGB feathering、latent compose 基线比较。

### M2：轻量 Joint Decoder

- 加入零初始化 residual fusion；
- 加入 2～3 个 shared ERP residual block；
- 使用 clean latent 完成阶段 A；
- 验证非接缝区域没有明显退化。

### M3：生成分布适配

- 缓存真实 TeleStyle/SphereAdapter latent 对；
- 完成阶段 B；
- 在未见场景、未见 style 和未见 seed 上评估。

### M4：联合优化

- 与 SphereAdapter 小规模联合微调；
- 尝试提前融合点或加入第二个多尺度融合点；
- 加入水平循环和可选极点拓扑 padding；
- 确认质量收益值得新增推理开销。

## 12. 成功判定

只有同时满足以下条件，才能认为版本二有效：

1. 赤道硬线和跨赤道梯度异常显著低于当前 hard-cut 基线；
2. 改善不是通过扩大模糊带实现，高频能量与边缘清晰度可接受；
3. 非赤道区域相对原 decoder 没有明显感知质量退化；
4. 未见场景和未见风格上仍有稳定收益；
5. 结构冲突严重时不会产生不可控伪影，且能识别该能力边界；
6. 显存和推理时间处于当前硬件可以接受的范围。

## 13. 结论

North/South Chart Joint Decoder 对当前 TeleStyle 是可行且直接的赤道接缝改进路线。最稳妥的实现不是重新训练球面 VAE，也不是第一版就引入完整多尺度 attention，而是：

> 保留现有 latent 与大部分 Qwen VAE decoder；用共享权重分别提取 North/South 中间 feature；将 feature 按真实球面 correspondence 重投影到统一 ERP；借鉴 SphereDiff，根据投影可靠度融合同一球面位置的 feature；再由零初始化残差模块修正，最后通过 shared ERP decoder 与单一 RGB head 重新生成完整全景图。

SphereDiff 最值得迁移的不是 Fibonacci 点云表示，而是“根据投影可靠度融合同一球面位置”的原则。把这个原则用于 North/South decoder feature，再让共享 ERP decoder 重新生成 RGB，比直接采用 SphereDiff 的最终多 patch RGB blending 更适合当前 TeleStyle。

项目成败的关键不仅是网络结构，还包括真实 TeleStyle 生成 latent 对的数据覆盖、训练/推理 geometry 完全一致，以及同时约束赤道连续性和非接缝区域保真度。

## 14. 参考资料

- SphereDiff 论文：<https://ojs.aaai.org/index.php/AAAI/article/download/37779/41741>
- SphereDiff 官方实现：<https://github.com/pmh9960/SphereDiff>
