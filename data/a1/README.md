# A1 ERP 数据目录

请将高质量、连续的 2:1 ERP 原图放到 `erp/`。目录支持递归扫描；建议每个
原始场景使用独立子目录，同一场景的不同版本放在同一子目录，避免跨数据集泄漏。

```text
data/a1/
├── erp/                 # 原始 ERP，不提交 Git
├── manifests/           # 自动生成的数据清单
├── cache/               # VAE latent、条件和几何缓存
└── stress/              # 程序生成的高频压力样本
```

放入数据后执行：

```bash
python -m training.prepare_a1_data --config configs/a1.yaml
```

工具只读取原图；无效图片会写入报告，不会修改或删除源文件。正式训练前会估算
缓存空间，空间不足时直接停止。
