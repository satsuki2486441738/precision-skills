# 仓库维护与归属

- 通用核心位于 `scripts/compare_scores.py`，永久测试在 `tests/`，架构与诊断规范在 `docs/`；诊断入口见 [SKILL.md](SKILL.md)。
- `projects/<project>/environments/` 独立管理项目环境声明；每个 case 的双侧 manifest 分别引用自身环境配置，不能以配置替代运行证据。
- `projects/<project>/cases/<case>/` 管理契约、manifest、README、WORK_LOG、专用 scripts、tests 和 runs。工作记录归案例，工具测试不等于案例实际执行证据。
- 脚本和测试按实际复用范围归属：案例专用就近放置，多个 case 共用才上提至项目的 `scripts/`、`tests/`，跨项目通用能力归框架；不预建空目录。
- 每个 case 仅维护一份 README（使用说明与当前产物入口）和一份 WORK_LOG（时间线与 run 引用）；run 内只存运行产物，不复制案例文档。环境 README 独立保留。
- run 命名为 `<side>-<task>-<UTC时间戳YYYYmmddTHHMMSSZ>`：side 为 gpu/npu/compare，task 为稳定小写短横线名称，时间取执行启动时间；不混入 schema 版本或成功失败状态。独占创建目录，重名不覆盖。
- 模型和源数据保持外部位置，不随通用工具分发。历史 run 不改写、不覆盖；无可靠启动时间的旧产物迁移命名为 `<side>-<task>-migrated-<实际UTC迁移时间>`，时间不代表模型启动。逐文件保留字节和内部布局，WORK_LOG 记录路径映射，不建兼容链接；新验收另存报告，历史记录内路径不改写。
- 遵循 [WORKFLOW](docs/WORKFLOW.md) 的授权与判定边界。修改前确认方案，每批最多 3 文件；修 bug 先临时复现，修复后回归并删除临时测试，不删除永久测试。不擅自安装、联网、扩设备/预算、改变比较语义或补造缺失证据。
- 从仓库根运行 `python -B -m unittest discover -s tests -v`；文档变更检查本地链接与锚点。自比较不能当作跨设备通过。
